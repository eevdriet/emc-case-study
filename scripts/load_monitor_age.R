# Load information for both worms
ascaris = readRDS("../data/timeline_db_main_ascaris.rds")
hookworm = readRDS("../data/timeline_db_main_hookworm.rds")

n_simulations = 1000
n_scenarios = 16 # scenarios 17-24 are not monogeneic

# Extract all survey information for each of the worms
extract_surveys <- function(df) {
    epi_survey <- data.frame()

    for (scenario in 1:n_scenarios) {
        for (simulation in 1:n_simulations) {
            print(sprintf("Scenario %d:  %d", scenario, simulation))

            df[[simulation]][[scenario]]$monitor_age['scen'] = scenario
            df[[simulation]][[scenario]]$monitor_age['sim'] = simulation

            epi_survey <- rbind(epi_survey, df[[simulation]][[scenario]]$monitor_age)
        }
    }

    return(epi_survey)
}

ascaris_monitor_age = extract_surveys(ascaris)
hookworm_monitor_age = extract_surveys(hookworm)

# Export monitor age information
monitor_age = ascaris_monitor_age
save(monitor_age, file = "ascaris_monitor_age.rds")

monitor_age = hookworm_monitor_age
save(monitor_age, file = "hookworm_monitor_age.rds")
