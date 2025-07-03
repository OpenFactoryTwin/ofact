from pathlib import Path
import dill as pickle

from ofact.planning_services.model_generation.persistence import deserialize_state_model
from ofact.settings import ROOT_PATH

pkl_path = Path(str(ROOT_PATH).split("ofact")[0], 'projects/Schmaus/data/raw_dt/test.pkl')
digital_twin = deserialize_state_model(source_file_path=pkl_path)

class SmallDT:

    def __init__(self):
        self.features = digital_twin.features
        self.feature_clusters = digital_twin.feature_clusters
        self.order = digital_twin.get_orders()

pickle_path = "test.pkl"
with open(pickle_path, 'wb') as outp:
    pickle.dump(SmallDT(), outp, pickle.HIGHEST_PROTOCOL)


# Feature(external_identifications={'Schmaus': ['Standard Einlagerung'], 'static_model': ['_Standard_e_f']}, name='Standard', is_not_chosen_option=None, feature_cluster="feature_cluster", price=None, selection_probability_distribution': <ofact.twin.state_model.probabilities.SingleValueDistribution object at 0x0000022A5AA0FDA0>}
# Feature(external_identifications={'Schmaus': ['Standard Kommissionierung'], name: 'Standard', is_not_chosen_option=None, feature_cluster="feature_cluster", price=None, selection_probability_distribution': <ofact.twin.state_model.probabilities.SingleValueDistribution object at 0x0000022A5AA0FE00>}
# Feature(external_identifications={'Schmaus': ['B'], 'static_model': ['_B_f']}, name='B Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=1.0, selection_probability_distribution="prob")
# Feature(external_identifications={'Schmaus': ['F'], 'static_model': ['_F_f']}, name='F Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=1.0, selection_probability_distribution="prob")
# Feature(external_identifications={'Schmaus': ['H'], 'static_model': ['_H_f']}, name='H Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=1.0, selection_probability_distribution="prob")
# Feature(external_identifications={'Schmaus': ['I'], 'static_model': ['_I_f']}, name='I Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=1.0, selection_probability_distribution="prob")
# Feature(external_identifications={'Schmaus': ['K'], 'static_model': ['_K_f']}, name='K Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=1.0, selection_probability_distribution="prob")
# Feature(external_identifications={'Schmaus': ['L'], 'static_model': ['_L_f']}, name='L Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=1.0, selection_probability_distribution="prob")
# Feature(external_identifications={'Schmaus': ['N'], 'static_model': ['_N_f']}, name='N Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=1.0, selection_probability_distribution="prob")
# Feature(external_identifications={'Schmaus': ['Sonstige'], 'static_model': ['_Sonstige_f']}, name='Sonstige Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=1.0, selection_probability_distribution': <ofact.twin.state_model.probabilities.NormalDistribution object at 0x0000022A5AA342F0>}
# Feature(external_identifications={'Schmaus': ['Z'], 'static_model': ['_V_f']}, name='V Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=1.0, selection_probability_distribution': <ofact.twin.state_model.probabilities.NormalDistribution object at 0x0000022A5AA34380>}
# Feature(external_identifications={'Schmaus': ['Einlagern_B'], 'static_model': ['_Einlagern_B_f']}, name='B Einlager-Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=0.5, selection_probability_distributio="prob")
# Feature(external_identifications={'Schmaus': ['Einlagern_F'], 'static_model': ['_Einlagern_F_f']}, name='F Einlager-Artikel', is_not_chosen_option=None, feature_cluster="feature_cluster", price=0.5, selection_probability_distributio="prob")
