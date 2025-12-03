pipeline {
    agent any

    environment {
        VENV = "venv"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Set Up Python Environment') {
            steps {
                sh """
                    python3 -m venv ${VENV}
                    . ${VENV}/bin/activate
                    pip install --upgrade pip
                    pip install ultralytics
                """
            }
        }

        stage('Run YOLO Model') {
            steps {
                sh """
                    . ${VENV}/bin/activate
                    yolo predict model=best.pt source=images/ save=True name=jenkins_output
                """
            }
        }

        stage('Evaluate Accuracy (Placeholder)') {
            steps {
                sh """
                    . ${VENV}/bin/activate
                    python accuracy_eval_placeholder.py --predictions runs/detect/jenkins_output --output metrics.json
                """
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'runs/detect/jenkins_output/**', fingerprint: true
            archiveArtifacts artifacts: 'metrics.json', fingerprint: true
        }
    }
}
