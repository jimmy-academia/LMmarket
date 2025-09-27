## task description to codex agent

- Goal: process user location and distance exponential decay factor parameter
- target script: `data/process_yelp.py` 
- target function to implement: `process_yelp_data`
- input: `args`, `DATA` (created by `DATA = load_or_build(args.prepared_data_path, dumpj, loadj, prepare_data, args)` in `main.py`)
- output: `USER_loc` (for each user, give user location and exponent parameter)

### description

we are going to use exponential distance decay (see `doc/distance.md`)

**For each user (within a city):**

1. Collect all restaurant locations they reviewed.
2. If the user has fewer than a minimum number of reviews (e.g., 5), skip them entirely.
3. Estimate the **user center** as the geometric median of those restaurant coordinates.
4. Compute all distances from this center to the reviewed restaurants.
5. Fit the exponential decay coefficient as (\lambda = 1 / \text{mean distance}).
6. Save for this user:

   * `center_lat`, `center_lon`
   * `lambda_per_km`

That’s it — one point (location) and one parameter (decay rate) per user.

Do you want me to draft a short comment-style description (like docstring text) you can drop directly at the top of `process_yelp.py` to guide the implementation?
