# Load information for both worms
ascaris = readRDS("../data/timeline_db_main_ascaris.rds")
hookworm = readRDS("../data/timeline_db_main_hookworm.rds")
scenarios = load("../data/scenarios_main.RData")

n_simulations = 1000
n_scenarios = 16 # scenarios 17-24 are not monogeneic

relevant_cols = c("time", "age_cat", "n_host", "n_host_eggpos", "a_epg_obs")

# Extract all survey information for each of the worms
extract_surveys <- function(df) {
    name <- deparse(substitute(df))

    for (scenario in 1:n_scenarios) {
        for (simulation in 1:n_simulations) {
            print(sprintf("Scenario %d:  %d", scenario, simulation))
            
            df_inner <- df[[simulation]][[scenario]]$monitor_age
            df_inner <- subset(df_inner, select=relevant_cols)
            
            df_inner$scenario <- scenario
            df_inner$simulation <- simulation
            
            file <- sprintf("../csv/%s_monitor_ageSC%02dSIM%04d.csv", name, scenario, simulation)
            write.csv(df_inner, row.names = FALSE, file = file)
        }
    }
}

extract_surveys(ascaris)
extract_surveys(hookworm)
