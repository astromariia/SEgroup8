pipeline {
    agent any

    environment {
        VENV = "venv"
    }

    stages {

        stage('Checkout Repository') {
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
                    pip install -r requirements.txt || true
                """
            }
        }

        stage('Run Model Test Script') {
            steps {
                sh """
                    . ${VENV}/bin/activate
                    python modelTest.py
                """
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/*.txt', fingerprint: true
            archiveArtifacts artifacts: '**/*.json', fingerprint: true
            archiveArtifacts artifacts: '**/*.png', fingerprint: true
        }
    }
}
