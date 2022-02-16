format:
	isort gridverse_experiment.py flat_pomdp_experiment.py online_pomdp_planning_experiments tests
	black gridverse_experiment.py flat_pomdp_experiment.py online_pomdp_planning_experiments tests

lint:
	flake8 gridverse_experiment.py flat_pomdp_experiment.py online_pomdp_planning_experiments tests
	pylint gridverse_experiment.py flat_pomdp_experiment.py online_pomdp_planning_experiments tests

test:
	pytest tests
