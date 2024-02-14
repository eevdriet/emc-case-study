# Load information for both worms
ascaris <- readRDS("../data/timeline_db_main_ascaris.rds")
hookworm <- readRDS("../data/timeline_db_main_hookworm.rds")

n_simulations <- 1000
n_scenarios <- 16

# Extract all survey information for each of the worms
extract_surveys <- function(df) {
  name <- deparse(substitute(df))
  for (scenario in 1:n_scenarios) {
    for (simulation in 1:n_simulations) {
      output_df <- data.frame(simulation = integer(),
                              scenario = integer(),
                              treat_time = numeric(),
                              host = numeric(),
                              pre = numeric(),
                              post = numeric())
      
      print(sprintf("Scenario %d: %d", scenario, simulation))
      
      de_survey <- df[[simulation]][[scenario]]$drug_efficacy
      required_columns <- c('treat_time', 'host', 'pre', 'post')
      
      # Check if all required columns are present
      missing_columns <- setdiff(required_columns, names(de_survey))
      
      # If any columns are missing, add them with NaN values
      if (length(missing_columns) > 0) {
        for (col in missing_columns) {
          de_survey[[col]] <- NaN
        }
      }
      
      de_survey_df <- data.frame(scenario = scenario, simulation = simulation, drug_efficacy = de_survey)
      
      file <- sprintf("../csv/%s_drug_efficacySC%02dSIM%04d.csv", name, scenario, simulation)
      write.csv(de_survey_df, row.names = FALSE, file = file)
    }
    
    
  }
}

extract_surveys(ascaris)
extract_surveys(hookworm)
