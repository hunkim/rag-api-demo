VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3
STREAMLIT = $(VENV)/bin/streamlit

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

chat: $(VENV)/bin/activate
	$(STREAMLIT) run chat.py 

autochat: $(VENV)/bin/activate
	$(STREAMLIT) run autochat.py 

benchmark: $(VENV)/bin/activate
	$(STREAMLIT) run benchmark.py

put_files: $(VENV)/bin/activate
	$(PYTHON) put_files.py

txt2pdf: $(VENV)/bin/activate
	$(PYTHON) txt2pdf.py

clean:
	rm -rf __pycache__
	rm -rf $(VENV)