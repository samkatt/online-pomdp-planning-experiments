format:
	isort *.py online_pomdp_planning_experiments tests
	black *.py online_pomdp_planning_experiments tests

lint:
	flake8 *.py online_pomdp_planning_experiments tests
	pylint *.py online_pomdp_planning_experiments tests

test:
	pytest tests
