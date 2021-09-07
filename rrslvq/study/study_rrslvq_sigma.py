
from __future__ import division

from joblib import Parallel, delayed
from rrslvq.utils.reoccuring_drift_stream import ReoccuringDriftStream
from skmultiflow.prototype.robust_soft_learning_vector_quantization import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.lazy import KNN
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from rrslvq.rrslvq_cls import ReactiveRobustSoftLearningVectorQuantization as RRSLVQ
from rrslvq.utils.reoccuring_drift_stream import ReoccuringDriftStream
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.lazy.sam_knn import SAMKNN
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from rrslvq.utils.study import Study
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.sea_generator import SEAGenerator
def init_classifiers():
    n_prototypes_per_class = 4
    sigma = 4
    rslvq = RSLVQ(prototypes_per_class=4, sigma=4)
    rrslvq = RSLVQ(prototypes_per_class=n_prototypes_per_class, sigma=sigma, confidence=0.0001, window_size=300)


    clfs = [rrslvq,rslvq]
    names = ["rrslvq","rslvq"]
    return clfs,names

def evaluate(stream,metrics,study_size):
    clfs,names = init_classifiers()
    stream.prepare_for_use()
    evaluator = EvaluatePrequential(show_plot=False, batch_size=10, max_samples=study_size, metrics=metrics,
                                    output_file=stream.name+" "+str(names)+".csv")

    evaluator.evaluate(stream=stream, model=clfs, model_names=names)

s = Study()
parallel = 8
study_size = 100000
metrics = ['accuracy',"kappa"]

streams = s.init_standard_streams()  + s.init_reoccuring_standard_streams() + s.init_real_world()

Parallel(n_jobs=parallel,max_nbytes=None)(delayed(evaluate)(stream,metrics,study_size) for stream in streams)
#