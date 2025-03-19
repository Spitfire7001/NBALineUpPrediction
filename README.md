# NBA Lineup Prediction Project
The overall purpose of this project is to predict an optimal 5th member for the home team in an NBA game. 
In order to achieve this goal, our group leverged label encoding and a random forest classifer.
More information can be found in the report present in this repositorty at this [link](https://github.com/Spitfire7001/NBALineUpPrediction/blob/main/NBA%20Line%20Prediction%20Report%20-%20Group%201.pdf).
The following information will be the process to run this model yourself in your own python virtual environment.

## Installation Procedure

> [!NOTE]
> Setup will slightly vary if you are using Linux or Windows. This will be made clear.

To start, make a new directory where the repo can be cloned in to

```
mkdir yourDirName
cd yourDirName
```

Next, clone the repo into the directory that was just created
```
git clone https://github.com/Spitfire7001/NBALineUpPrediction.git
```

Now a Python VENV will be created so the dependencies will not be installed system wide
```
python -m venv .venv
```
> [!IMPORTANT]
> This is where Windows and Linux Setup Slightly Varies

On Linux use:
```
source .venv/bin/activate
```
On Windows in Command Prompt:
```
.venv\Scripts\activate
```
Or on Windows in PowerShell
```
.\.venv\Scripts\Activate.ps1
```

Once the Virtual Environment is created and activated, the dependencies can be installed
```
pip install pandas numpy scikit-learn
```

Once all these steps are completed setup is done.

## Running Procedure
The data provided in the repo is already preprocessed and the model is ready to be run. If you want to add you own data the steps will be as follows:

Place the test data into the test_data directory. After run:
```
python correctTeamNames.py
```
And
```
python splitTraining.py
```
Once these have completed placed the resulting files in the proper test_data and training_data directory.
Now that the data is prepped you can run:
```
python NBAPredict.py
```
to start the model. The results will be outputted to **results.csv** in the current directory.

