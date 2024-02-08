# Load all metadata from the scenarios / simulations
library(jsonlite)

# Load information for both worms
ascaris = readRDS("../data/timeline_db_main_ascaris.rds")
hookworm = readRDS("../data/timeline_db_main_hookworm.rds")

n_obs = 1000
n_scenarios = 16 # scenarios 17-24 are not monogeneic

# Aggregate all metadata information for the scenarios and simulations
ascaris_metadata <- list()
hookworm_metadata <- list()

for (scenario in 1:n_scenarios) {
    ascaris_metadata[[scenario]] = list()
    hookworm_metadata[[scenario]] = list()
}

for (scenario in 1:n_scenarios) {
    # Get scenario information
    ascaris_metadata[[scenario]] = c(ascaris_metadata[[scenario]], ascaris[[1]][[scenario]]$scen)
    hookworm_metadata[[scenario]] = c(hookworm_metadata[[scenario]], hookworm[[1]][[scenario]]$scen)
    
    # Get simulations information
    ascaris_sims = list()
    hookworm_sims = list()
    for (obs in 1:n_obs) {
        ascaris_sims = c(ascaris_sims, list(ascaris[[obs]][[scenario]]$par_input))
        hookworm_sims = c(hookworm_sims, list(hookworm[[obs]][[scenario]]$par_input))
    }
    
    ascaris_metadata[[scenario]]$simulations = ascaris_sims
    hookworm_metadata[[scenario]]$simulations = hookworm_sims
}
    
# Export metadata
json = jsonlite::toJSON(ascaris_metadata, pretty = TRUE, auto_unbox = TRUE)
writeLines(json, "ascaris_metadata.json")

json = jsonlite::toJSON(hookworm_metadata, pretty = TRUE, auto_unbox = TRUE)
writeLines(json, "hookworm_metadata.json")