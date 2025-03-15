
from __future__ import division
import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../../"))
from joblib import Parallel, delayed
from rrslvq.utils.reoccuring_drift_stream import ReoccuringDriftStream
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.lazy import KNN
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from rrslvq.rrslvq_cls import ReactiveRobustSoftLearningVectorQuantization
from rrslvq.utils.reoccuring_drift_stream import ReoccuringDriftStream
from rrslvq.rrslvq_cls import ReactiveRobustSoftLearningVectorQuantization as RRSLVQ
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from rrslvq.utils.study import Study
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.sea_generator import SEAGenerator
def init_classifiers():
    samknn = SAMKNN()
    clfs = [samknn]
    names = ["SamKnn"]
    return clfs,names

def evaluate(stream,metrics,study_size):
    clfs,names = init_classifiers()
    stream.prepare_for_use()
    evaluator = EvaluatePrequential(show_plot=False, batch_size=10, max_samples=study_size, metrics=metrics,
                                    output_file=stream.name+"_memory_other.csv")

    evaluator.evaluate(stream=stream, model=clfs, model_names=names)

s = Study()
parallel =2
study_size = 100000 
metrics = ['accuracy','model_size']

streams = s.init_standard_streams()  + s.init_reoccuring_standard_streams() + s.init_real_world()

Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(stream,metrics,study_size) for stream in streams)