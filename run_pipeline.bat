@echo off
call .venv\Scripts\activate
python scripts\load_data.py
python scripts\append_weather.py
python scripts\filter_stations.py
python scripts\feature_selector.py
python scripts\add_time_features.py
python scripts\scale_data.py
python src\training\prepare_dataset.py
pause
