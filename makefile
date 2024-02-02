
activate:
	@echo "Activating virtual environment"
	@python3 -m venv venv
	@source venv/bin/activate

install:
	@echo "Installing requirements"
	@python3 -m pip install -r requirements.txt

run:
	@echo "Running the application"
	@streamlit run app.py