from attrs import define
from emc.model.policy import Policy

@define
class Policy_Manager:
    # TODO: 
    # Create initial policy
    # 

    initialPolicy = Policy()
    
    # train / test split simulations per scenario, label using target

    # classification or regression

    # construct epi_survey schedule 
    # generate all subpolicies
    # generate all classifier models per subpolicy train on train set, test on test set

    # totalCosts = 0
    # per simulation in test set:
    #   from generated classifiers construct de_survey schedule
    #   for every time t from t = 0 to t = n
    #       if epi_survey[t]:
    #           totalCosts += epi_survey_costs at time t
    #       if de_survey[t]:
    #           totalCosts += de_survey_costs at time t
    #           if de_survey result < 0.85:
    #               continue
    #   if (de_efficacy < 0.85 but not found):
    #       totalCosts += 100000 (costs if not found)
    # averageCosts = totalCosts / len(simulation in test_set)

    # TODO: hoe goed is de policy = averageCosts


