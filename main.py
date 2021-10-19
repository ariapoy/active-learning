# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run active learner on classification tasks.

Supported datasets include mnist, letter, cifar10, newsgroup20, rcv1,
wikipedia attack, and select classification datasets from mldata.
See utils/create_data.py for all available datasets.

For binary classification, mnist_4_9 indicates mnist filtered down to just 4 and
9.
By default uses logistic regression but can also train using kernel SVM.
2 fold cv is used to tune regularization parameter over a exponential grid.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
from time import gmtime
from time import strftime
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
from numpy.lib.function_base import trim_zeros
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import pandas as pd

import tensorflow.compat.v1.gfile as gfile

from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import get_wrapper_AL_mapping
from utils import utils

from utils import evaluation

import pdb

# development
dev_mode = False
# data
normalize_data = "False"
standardize_data = "False"
# model
modelname = "kernel_svm"
svm_auto = "True"
svm_CV = "False"
# query strategy
qsname = "uniform"
batch_size_query = 2
# experiment
num_labeled = 20
min_num_labeled_perClass = 1
ratio_test = 0.4
train_valid_test_ratio_list = [1 - ratio_test, 0, ratio_test]
num_trials = 100

# adapative for Google AL toolbox
sampling_method = qsname
warmstart_size = num_labeled
batch_size = batch_size_query
trials = num_trials
seed = 0
confusions = "0."
confusions = confusions.split(" ")
active_sampling_percentage = "1.0"
active_sampling_percentage = active_sampling_percentage.split(" ")
score_method = modelname
select_method = "None"
max_dataset_size = "15000"
train_horizon = 1.0
data_dir = "./tmp/data"

get_wrapper_AL_mapping()

def generate_one_curve(X,
                       y,
                       sampler,
                       score_model,
                       seed,
                       warmstart_size,
                       batch_size,
                       select_model=None,
                       confusion=0.,
                       active_p=1.0,
                       max_points=None,
                       standardize_data=False,
                       norm_data=False,
                       train_horizon=0.5):
  """Creates one learning curve for both active and passive learning.

  Will calculate accuracy on validation set as the number of training data
  points increases for both PL and AL.
  Caveats: training method used is sensitive to sorting of the data so we
    resort all intermediate datasets

  Args:
    X: training data
    y: training labels
    sampler: sampling class from sampling_methods, assumes reference
      passed in and sampler not yet instantiated.
    score_model: model used to score the samplers.  Expects fit and predict
      methods to be implemented.
    seed: seed used for data shuffle and other sources of randomness in sampler
      or model training
    warmstart_size: float or int.  float indicates percentage of train data
      to use for initial model
    batch_size: float or int.  float indicates batch size as a percent of
      training data
    select_model: defaults to None, in which case the score model will be
      used to select new datapoints to label.  Model must implement fit, predict
      and depending on AL method may also need decision_function.
    confusion: percentage of labels of one class to flip to the other
    active_p: percent of batch to allocate to active learning
    max_points: limit dataset size for preliminary
    standardize_data: wheter to standardize the data to 0 mean unit variance
    norm_data: whether to normalize the data.  Default is False for logistic
      regression.
    train_horizon: how long to draw the curve for.  Percent of training data.

  Returns:
    results: dictionary of results for all samplers
    sampler_states: dictionary of sampler objects for debugging
  """
  # TODO(lishal): add option to find best hyperparameter setting first on
  # full dataset and fix the hyperparameter for the rest of the routine
  # This will save computation and also lead to more stable behavior for the
  # test accuracy

  # TODO(lishal): remove mixture parameter and have the mixture be specified as
  # a mixture of samplers strategy
  def select_batch(sampler, uniform_sampler, mixture, N, already_selected,
                   **kwargs):
    n_active = int(mixture * N)
    n_passive = N - n_active
    kwargs["N"] = n_active
    kwargs["already_selected"] = already_selected
    batch_AL = sampler.select_batch(**kwargs)
    already_selected = already_selected + batch_AL
    kwargs["N"] = n_passive
    kwargs["already_selected"] = already_selected
    batch_PL = uniform_sampler.select_batch(**kwargs)
    return batch_AL + batch_PL

  np.random.seed(seed)
  data_splits = train_valid_test_ratio_list

  if max_points is None:
    max_points = len(y)
  train_size = int(min(max_points, len(y)) * data_splits[0])
  if batch_size < 1:
    batch_size = int(batch_size * train_size)
  else:
    batch_size = int(batch_size)
  if warmstart_size < 1:
    # Set seed batch to provide enough samples to get at least 4 per class
    # TODO(lishal): switch to sklearn stratified sampler
    seed_batch = int(warmstart_size * train_size)
  else:
    seed_batch = int(warmstart_size)
  seed_batch = max(seed_batch, 6 * len(np.unique(y)))

  indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise = (
      utils.get_train_val_test_splits(X,y,max_points,seed,confusion,
                                      seed_batch, split=data_splits,
                                      least_num_obs_of_each_class=min_num_labeled_perClass))

  # Preprocess data
  if norm_data:
    # print("Normalizing data")
    X_train = normalize(X_train)
    X_val = normalize(X_val)
    X_test = normalize(X_test)
  if standardize_data:
    # print("Standardizing data")
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    if X_val.shape[0] > 0:
      X_val = scaler.transform(X_val)
    else:
      pass
    X_test = scaler.transform(X_test)
  y_train = y_train
  y_val = y_val
  y_test = y_test

  # print("active percentage: " + str(active_p) + " warmstart batch: " +
  #       str(seed_batch) + " batch size: " + str(batch_size) + " confusion: " +
  #       str(confusion) + " seed: " + str(seed))

  # Initialize samplers
  uniform_sampler = AL_MAPPING["uniform"](X_train, y_train, seed)
  sampler = sampler(X_train, y_train, seed)

  results = {}
  data_sizes = []
  accuracy = []
  # initial label
  selected_inds = list(range(seed_batch))

  # If select model is None, use score_model
  same_score_select = False
  if select_model is None:
    select_model = score_model
    same_score_select = True

  n_batches = int(np.ceil((train_horizon * train_size - seed_batch) *
                          1.0 / batch_size)) + 1
  for b in range(n_batches):
    n_train = seed_batch + min(train_size - seed_batch, b * batch_size)
    # print("Training model on " + str(n_train) + " datapoints")

    assert n_train == len(selected_inds)
    data_sizes.append(n_train)

    # Sort active_ind so that the end results matches that of uniform sampling
    partial_X = X_train[sorted(selected_inds)]
    partial_y = y_train[sorted(selected_inds)]
    score_model.fit(partial_X, partial_y)
    if not same_score_select:
      select_model.fit(partial_X, partial_y)
    acc = score_model.score(X_test, y_test)
    accuracy.append(acc)
    # print("Sampler: %s, Accuracy: %.2f%%" % (sampler.name, accuracy[-1]*100))

    n_sample = min(batch_size, train_size - len(selected_inds))
    select_batch_inputs = {
        "model": select_model,
        "labeled": dict(zip(selected_inds, y_train[selected_inds])),
        # "eval_acc": accuracy[-1],
        # "X_test": X_val,
        # "y_test": y_val,
        "y": y_train
    }
    new_batch = select_batch(sampler, uniform_sampler, active_p, n_sample,
                             selected_inds, **select_batch_inputs)
    selected_inds.extend(new_batch)
    # print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
    assert len(new_batch) == n_sample
    assert len(list(set(selected_inds))) == len(selected_inds)

  # Check that the returned indice are correct and will allow mapping to
  # training set from original data
  assert all(y_noise[indices[selected_inds]] == y_train[selected_inds])
  results["accuracy"] = accuracy
  results["selected_inds"] = selected_inds
  results["data_sizes"] = data_sizes
  results["indices"] = indices
  results["noisy_targets"] = y_noise
  return results, sampler

def table3_summary(aubc_arr, result_tag=0, verbose=0):
    result_tag_map = {0: "BSO", 1: "RS", 2: "US"}
    mean_aubc = np.mean(aubc_arr)
    mse_aubc = np.std(aubc_arr) / np.sqrt(aubc_arr.shape[0])
    if verbose:
        print("Table 3. {1} results {0:.3f}".format(mean_aubc, result_tag_map[result_tag]))
        print("Confidence interval of Table 3. {2} results ({0:.3f}, {1:.3f})".format(mean_aubc - 2*mse_aubc, mean_aubc+ 2*mse_aubc, result_tag_map[result_tag]))
        print("\t{1}/{0}".format(num_trials, aubc_arr.shape[0]))  # Check how many experiments we get.
    return "{0:.3f}/{1:.4f}".format(mean_aubc, mse_aubc)

def OLHC(lc):
    open_value, open_idx = lc[0], 0
    low_value, low_idx = np.min(lc), np.argmin(lc)
    high_value, high_idx = np.max(lc), np.argmax(lc)
    close_value, close_idx = lc[-1], lc.shape[0]
    results_tuple = (open_value, open_idx, low_value, low_idx, high_value, high_idx, close_value, close_idx)
    return "{0:.3f}({1})/{2:.3f}({3})/{4:.3f}({5})/{6:.3f}({7})".format(*results_tuple)

# main
# datanames_list = ["iris", "wine", "sonar", "seeds", "glass", "thyroid", "heart", "haberman", "ionosphere", "clean1", "wdbc", "australian", "diabetes", "vehicle", "german.numer"]
datanames_list = ["iris", "wine", "sonar", "glass", "heart", "ionosphere", "australian", "diabetes", "vehicle", "german"]
report = {"dataset": [], "AUBC google/active-learning.RS": []}
report2 = {"dataset": [], "OLHC google/active-learning.RS": []}

def run(seed):
    # Initialize models
    sampler = get_AL_sampler(sampling_method)
    global svm_CV
    svm_CV = svm_CV == "True"
    score_model = utils.get_model(score_method, seed, is_gridsearch=svm_CV)
    if (select_method == "None" or
        select_method == score_method):
        select_model = None
    else:
        select_model = utils.get_model(select_method, seed)

    results, sampler_state = generate_one_curve(
        X, y, sampler, score_model, seed, warmstart_size,
        batch_size, select_model, c, m, max_dataset_size,
        standardize_data, normalize_data, train_horizon)
    key = (dataset, sampling_method, score_method,
            select_method, m, warmstart_size, batch_size,
            c, standardize_data, normalize_data, seed)
    sampler_output = sampler_state.to_dict()
    results["sampler_output"] = sampler_output
    return key, results


for dataname in tqdm(datanames_list):
    dataset = dataname
    confusions = [float(t) for t in confusions]
    mixtures = [float(t) for t in active_sampling_percentage]
    all_results = {}
    max_dataset_size = None if max_dataset_size == "0" else int(
        max_dataset_size)
    normalize_data = normalize_data == "True"
    standardize_data = standardize_data == "True"
    X, y = utils.get_mldata(data_dir, dataset, src="zhan")
    starting_seed = seed

    for c in confusions:
        for m in mixtures:
            if dev_mode:
                for seed in range(3):
                    key_curr, results_curr = run(seed)
                    all_results[key_curr] = results_curr
            else:
                with Pool(6) as p:
                    imap_unordered_it = p.imap_unordered(run,
                                            range(starting_seed, starting_seed + trials)
                                        )
                    for res in imap_unordered_it:
                        all_results[res[0]] = res[1]

    table3_lcs_RS = []
    for exp_key in all_results:
        table3_lcs_RS.append(all_results[exp_key]["accuracy"])

    table3_lcs_RS = np.array(table3_lcs_RS)
    num_queries = np.arange(num_labeled, num_labeled + table3_lcs_RS[0].shape[0])
    table3_aubc_RS = np.array([evaluation.AUBC(num_queries, table3_lcs_RS[exp_id]) for exp_id in range(table3_lcs_RS.shape[0])])

    report["dataset"].append(dataname)
    report["AUBC google/active-learning.RS"].append(table3_summary(table3_aubc_RS, 1))

    report2["dataset"].append(dataname)
    mean_lc_RS = table3_lcs_RS.mean(axis=0)
    report2["OLHC google/active-learning.RS"].append(OLHC(mean_lc_RS))

reportAUBC = pd.DataFrame(report)
reportLCOLHC = pd.DataFrame(report2)

reportAUBC.to_csv("Table1-AUBC-Google-20211017-2.csv", index=None)
reportLCOLHC.to_csv("Table2-OLHC-Google-20211017-2.csv", index=None)
