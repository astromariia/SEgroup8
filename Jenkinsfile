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
                bat """
                    python -m venv %VENV%
                    call %VENV%\\Scripts\\activate
                    python -m pip install --upgrade pip
                    pip install ultralytics
                    if exist requirements.txt pip install -r requirements.txt
                """
            }
        }

        stage('Run Model Test Script') {
            steps {
                bat """
                    call %VENV%\\Scripts\\activate
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
