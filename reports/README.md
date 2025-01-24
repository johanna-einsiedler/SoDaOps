# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15) [used huggingface]

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [x] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [x] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [x] Write some documentation for your application (M32)
* [x] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
MLOps51


### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
>
> Answer:
- Johanna Einsiedler - s250338
- Magnus Nielsen - enrolment difficulties
- Marten Appel - enrolment difficulties

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the third-party framework HuggingFace in our project. We used functionality of downloading a pretrained transformer from HuggingFace Hub and finetuning it using their Trainer functionality.
These finetuned models are then used as our final models.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used pip and conda for managing our dependencies. The list of dependencies was auto-generated using pip freeze > requirements.txt . To get a complete copy of our development environment, one would have to run the following commands "conda create --name my_environment python=3.10" and "pip install -r requirements.txt"

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We used the cookiecutter template provided by the course. We added the following folders
- .dvc
- downloaded_model
These are used to track data using dvc and to contain the best downloaded model, whereas the model folder is used to save models during training.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

Yes, we used ruff to automatically format our codebase and make sure that the coding style is coherent across team members. Similarly, we tried to make sure to include documentation for all functions that we implemented. Especially in projects with multiple stakeholders this type of streamlining is important to make sure that collaboration can happen smoothly and no unnecessary time is wasted on trying to decipher other people's code. 

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented
- 2 test for data preprocessing, checking the data saving process and the data cleaning prcoess with mock data
- 1 test for the training, checking that the data loading works
- 1 test for the model, checking that the pipeline works
- 1 test for the api with a sample input

So in total we implemented 5 tests, asserting that the key functions work as expected. 

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The overall coerage is 13\%, partially also because we did not implement test for non-cores scripts such as the visualisation fo the model outcomes. However, also if the coverage would be close to 100\% one cannot be sure that everything works as expected. Unit test always only test very particular aspects of isolated parts of code. Thus, there can also occur errors that hadn't been anticipated (and thus not considered in the testing) or that occur when parts of the application are connected (which would need more integrative testing to be detected).

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Yes, our workflow did include branches but not pull requests. Initially, we all created personal branches to work on, which we mainly worked on and pushed to main. We originally implemented branch protection rules, but due to the rapid development pace at the beginning of a project, this was too cumbersome. However, especially for larger coding projects were it is much harder to maintain an overview of who is working on what, branches in combination with pull requests and a list of required test that need to be passed before a merge is enabled, are definitely helpful to make sure it doesn't get all messy. Due to GCP authorization, we created a forked GitHub repo where we implement CI and CI/CD pipelines with cloudbuild triggers on CI and CI/CD branches. Here we would create branch protection rules such as requiring review and passing tests before allowing merging.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC in our project by uploading the raw data to a public bucket. It did as such not help us in this specific project as we did not require updating the version, but in future cases it will be helpful as it allows to version the training/validation and test data used. As a hypothetical scenario, we log requests to our API and would be able to download these logs and annotate them, adding them to the raw data files in DVC. We could then train a new model with this extended data.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We use GitHub actions to test:
- Formatting
- Unit testing
- API testing
- Coverage

We only run tests on Linux as our Docker images also use Linux, using caching for pip to reduce test times. Triggered GitHub actions can be seen here <https://github.com/johanna-einsiedler/SoDaOps/actions>. 

Additionally we created a forked repo (due to GCP requiring that triggers are associated with repos owned by the user) 


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used Weights & Biases to run and log experiments. We used sweeps to optimize hyperparameters. All training was done using Vertex AI, and would be triggered using:

`gcloud builds submit --config=cloudbuild_vertex.yaml .`

Which in turn runs a:

`wandb sweep config=configs/sweep.yaml`




### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We made use of config files and wandb. Hyperparameters which are optimized over are decided in the `sweep.yaml` file, with additional hyperparameters given in the environment `vertex_config_secret.yaml`, e.g., `num_train_epochs`, `max_length`, `batch_size`. These are logged to both log files and wandb logs.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

[WandB sweep](figures/wandb_sweep.PNG)
As seen in the first image we have used sweeps to select the best model using validation loss. This informs us about the best hyperparameters and how different constellations affect out of sample performance.

[WandB trainign loss](figures/train_loss.PNG)
As seen in the second image we are also tracking the training loss. This enables us to examine whether our model training is behaving as expected, with a smooth downward curve which is plateauing towards the end.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We used docker to build images for:

- Data preprocessing and training (main)
- API
- Data drift

We have primarily built and run these using Cloud build and Vertex AI, e.g., `gcloud builds submit --config=cloudbuild_vertex.yaml .` The main dockerfile is `main.dockerfiles` <https://github.com/johanna-einsiedler/SoDaOps/blob/main/dockerfiles/main.dockerfile>, which will run the data processing, training with upload to WandB and selection of best model with upload to Artifact registry.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging was dependent on gropup member. We mostly performed debugging by running the code and following up on error messages, as many error were from the cloud building process. Some members also used the debugger, primarily when working on the model pipeline and assessing shapes etc.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used:
- Cloud build: The backbone which runs the majority of our code.
- Cloud Storage: Used to create buckets that can contain data, models, etc.
- Vertex AI: Used to spin up VM's for training in combination with WANDB.
- Compute Engine: To build scripts initially for debugging purposes mainly.
- Artifact Registry: Used to build and store docker images in the cloud and retrieve if needed.
- Cloud Run: Used to deploy API's accessible from the web.
- Monitoring: To setup API alerts.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the Compute Engine to run docker images in the beginning for debugging purposes. However, we quickly changed to used Vertex AI and running models locally.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![Bucket](figures/bucket.PNG)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![Registry](figures/artifact_registry.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Build History](figures/cloud_build.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We managed to train our model using Vertex AI. The reason we did this is because they run the docker image and then they shut down automatically.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We did manage to write an API for our model. We used FastAPI to do this and loaded the best model trained using Vertex AI which we saved to the Cloud Storage, which was easily accesible through Cloud Run. Then we took an input string and returned an output to the user. Additionally we added logging save inputs such that we can annotate and extend training data later.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We used Cloud Run to deploy our model API, current version is <https://ci-cd-api-b04df250-aac9-4514-8aa0-4ebc08622611-727127387938.europe-west1.run.app>. It can be accessed using `/docs` or using `curl -X "GET" "https://ci-cd-api-b04df250-aac9-4514-8aa0-4ebc08622611-727127387938.europe-west1.run.app/predict?review=<YOUR TEXT HERE>"`

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We performed unit testing of our API using a hardcoded version. This asserts basic properties about the probabilities outputted. We would like to add load testing and ensuring that the API which was deployed most recently is always tested.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We implemented monitoring of our deploymed model using GCP triggers. These ensure that a group member receives a mail if there are an excess amount of logs to Cloud Run overall (thus capturing all our deployed models). This group member could then follow up and examine the monitoring tab, seeing what services are causing excessive load and act on this.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

Marten used 109 dkk.
Johanna used xxx dkk.
Magnus used 45 dkk.
The service costing the most was Artifact Registry (Magnus) and Compute Engine (Marten)

We purposefully chose a very small dataset (train N = 500) and used only CPU, as had issues accessing GPUs due to ressource shortages during exercises.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We implemented a drift detection system that can run in the cloud. Currently it is only run on the validation set of our original data, but can hypothetically be used to detect drift in input to the api. The drift detection was implemented using evidently. 

We implemented logging of input queries to the API, to enable human annotation and later extension of the training data. 

We created two Cloud Build triggers on branches on a forked GitHub Repo (<https://github.com/Magnus-Nielsen/SoDaOps>): A CI trigger and a CI/CD trigger. These trigger Cloud Builds which either train a full model (CI) or train a full model and deploy to API (CI/CD) when there are pushes to these branches. 

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

[architecture](figures/architecture.png)


### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The biggest challenge in the project was to get docker images to work and to work in the cloud due to the long compile times making fixing errors a headache. Additionally, the Cloud Build syntax could cause some challenges when trying to chain multiple steps together.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Johanna was responsible for setting up Github, the API and testing.
Marten was responsible for setting up the initial python scripts, the dvc and create the data drift evaluation. 
Magnus was responsible for setting WandB tracking up, training on Vertex AI, migrating API to Cloud Run, creating Cloud Builds. 
