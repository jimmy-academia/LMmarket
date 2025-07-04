# LMmarket


## Plan

> Idea: we will down the following stage-by-stage, and back tracking when problem arise that needs a fix in previous stage.

### Stage 1: Prepare Dataset

> Repeat/Parallel development for Yelp and Amazon review

#### A: basic data preparation
1. Filter user and item list
2. Consolidate central theme for static feature.
3. Complete user and item profile based on the central theme for static feature
    - {numerical + text descriptive} for static feature
    - price estimations: user = budget and price sensitivity; item = margin cost

#### B: advanced data synthesization
4. Prepare full table of dynamic hard constraints => (label, selection range, description; relevant search keyword). Backtrack update with step 5
5. For each user, synthesize request:
    - prepare prompt and prompt engineering algorithm (random selection on full table of dynamic preferences) for request generization.
        - generate request
        - validate result 0: make sure it is aligned with the user profile
        - validate result 1: make sure it is aligned utilized dynamic preferences
        - validate result 2: search for the keywords dynamic preference (update relevant search keywords); validate relevant items exists in the dataset.
        - locate (top 10) reference ground truth (based only on feature, not on price)
6. for each item, prelabel hard constraints ... manual/llm assisted annotation... (concurrently with validate result 2)


### Stage 2: Prepare Evaluation Environment
7. Prepare coding environment:
    - train test split
    - iterate over request; given (top 5) item recommendation, calculate result:
        - filter hard constraint with prelabel
        - calculate user--item static utility (with numerical profile)
    - tally buyer utility and seller revenue results

### Stage 3: Adapt and implement baselines:
8. Work on LLM-ETL methods
9. Work on dense retrieval methods
10. Work on generative retrieval methods


------

## usages

### create dataset
`python -m dataset.make_yelp`
