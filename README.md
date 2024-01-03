# Samay AWS Ml Architecture
This is the repository for the development of the ML pipeline for Samay Health.
AI-driven system integrating a respiratory disease detection model into a clinical management platform on AWS, incorporating the considerations you mentioned:


![Arquitectura](https://raw.githubusercontent.com/victorhtorres/SoyInformatico/master/MatematicasDiscretas/Images/formula-suma-termino-progresion-geometrica.png)

**System Architecture:**

**Data Flow:**

**Data Acquisition:**
Lung sound recordings are collected using medical devices or smartphone apps.
Data is securely transmitted to AWS using Amazon API Gateway or Amazon Kinesis Data Streams.

**Data Preprocessing:**
Data is stored in Amazon S3 for long-term storage.

Amazon SageMaker Processing Jobs are used for preprocessing:
Cleaning and normalization.
Feature extraction.

**Model Inference:**
Preprocessed data is fed to the respiratory disease detection model deployed on Amazon SageMaker Endpoints.
Model generates predictions (disease classification, probability scores).

**Prediction Storage:**
Predictions are stored in Amazon S3 or Amazon DynamoDB.

**Integration with Clinical Platform:**
Predictions are integrated into the clinical management platform's database using appropriate APIs.
Results are displayed in the platform's user interface for clinical decision support.

**Model Integration:**

**Model Deployment:** The model is deployed as a SageMaker Endpoint, accessible via API calls.

**Integration with User Interface:** The clinical management platform's UI makes API calls to the SageMaker Endpoint to retrieve predictions and display results seamlessly.

I**ntegration with Database:** Predictions are stored in the platform's database for long-term tracking and analysis.

**Integration with Other Tools:** The system could potentially interact with other diagnostic tools or EHR systems, depending on the platform's capabilities.
Interaction with System Components:

**Scalability:** The system is designed to handle increasing data volumes and usage by leveraging AWS's scalable infrastructure.

**Security:** Sensitive data is protected using encryption, access controls

**User Accessibility:** The system provides a user-friendly interface for clinicians to access and interpret model results easily.
Challenges and Solutions:

**Data Quality:** Ensure high-quality data for model accuracy by implementing data validation and cleaning processes.

**Model Explainability:** Provide insights into model predictions for clinicians

**Integration Complexity:** Address integration challenges through careful planning, testing, and potentially using middleware or APIs for seamless communication.

**Regulatory Compliance:** Adhere to healthcare regulations like HIPAA and GDPR for data privacy and security.

**Additional Considerations:**

**Model Monitoring:** Continuously monitor model performance and retrain as needed to maintain accuracy.

**Feedback Mechanism:** Gather feedback from clinicians to improve the system's usability and effectiveness.

**Cost Optimization:** Leverage AWS's cost optimization tools and strategies to manage expenses.

##Incorporating Amazon QuickSight into the Architecture:

**Data Preparation:**

**Data Transformation**: If needed, transform data in S3 using AWS Glue or other data processing services to prepare it for visualization.

**QuickSight Dataset Creation:** Create datasets in QuickSight directly from S3 data sources.

**Dashboard Creation:**

**User Interactions:**

- Visualize user interactions with the app, such as:
- Number of recordings uploaded.
- Time spent on different app features.
- User feedback and ratings.
- Track trends and identify areas for improvement.

**Inferences and Model Evaluations:**

- Display model predictions and performance metrics over time.
- Monitor model accuracy and fairness.
- Identify potential biases and areas for retraining.

**Additional Data from S3:**

Integrate other relevant data from S3, such as:

- Patient demographics.
- Clinical outcomes.
- Cost data.
- Gain deeper insights into disease patterns, treatment effectiveness, and resource utilization.

**Benefits of Using QuickSight:**

- Interactive Visualizations: Create engaging and informative dashboards with a wide range of visualizations.

- Real-Time Data: Analyze data as it's updated in S3 for near real-time insights.

- Collaboration and Sharing: Easily share dashboards with stakeholders for collaboration and decision-making.

- Secure Access: Control access to dashboards using AWS IAM.
Integration with Other AWS Services: Integrate with services like Amazon SageMaker for insights into model training and performance.



## PredictionsSamayPipeline

Overview

This pipeline automates the training, evaluation, and deployment of a TensorFlow model on Amazon SageMaker. It uses EventBridge to trigger the pipeline when new data is available in a Kinesis Data Stream.

Key Components

1. EventBridge Rule:

Detects new data in the Kinesis Data Stream.
Triggers a Lambda function to execute the pipeline.
2. Lambda Function:

Starts a SageMaker pipeline execution.
3. SageMaker Pipeline:

Data Processing: Preprocesses data using SKLearnProcessor.

Model Training: Trains a TensorFlow model using TensorFlowEstimator.

Model Evaluation: Evaluates model performance using a custom ScriptProcessor.

Conditional Step: Registers and deploys the model only if accuracy meets a threshold (0.7).

Model Registration: Registers the model in the SageMaker Model Registry (if accuracy condition is met).

Model Deployment: Deploys the model for inference using TensorFlowProcessor (if accuracy condition is met).

Prerequisites

An AWS account with SageMaker and EventBridge permissions.

A Kinesis Data Stream with sample data.

Python scripts for data preprocessing, model training, evaluation, and prediction.

Usage

Deploy the Lambda function with necessary permissions.

Create the EventBridge rule and target the Lambda function.

Start the EventBridge rule.

Run the pipeline once manually to create necessary resources.

Add new data to the Kinesis Data Stream to trigger the pipeline automatically.

Additional Information

Parameters: The pipeline can be customized with parameters for instance types, instance count, model approval status, and input data source.
Conditional Step: Ensures model quality before deployment.

Model Metrics: Captured and associated with the registered model.

References

SageMaker Pipelines: https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html
EventBridge: https://docs.aws.amazon.com/eventbridge/latest/userguide/
Lambda: https://docs.aws.amazon.com/lambda/latest/dg/
