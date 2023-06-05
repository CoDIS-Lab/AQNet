# AQNet - Air Quality prediction Network

The AQNet implementation code for the paper entitled *Predicting air quality via multimodal AI and satellite imagery* will be made available here soon.

**Folders and their content can be given below:**
```
These folders contain the documents necessary for regenerating outputs.

\Checkpoints
	 - One pretrained ResNet50 network from previous project.
	 - One multimodal, MobileNet backboned model for testing OOS cases.
\Data
	\data
		\basic
		  - Contains the original supporting data from the StGallen resources.
		\editted
		  - And here are sample lookups and generator created during this project to facilitate multioutput predictions.
	\timeseries_data
	  - Outputs from running the "discomap_extractor.ipynb" notebook, aggregated pollution measurements.
\Modules
	+ Contains two subfolders, one for each iteration of testing/modelling.
	\SingleOutput
		- .py files for modelling with 1 pollutant with mulitmodal architecture
	\ThreeOutput
		- .py files for modelling with 3 pollutants with mulitmodal architecture
```
