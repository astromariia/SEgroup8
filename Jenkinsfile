pipeline {
    agent any

    environment {
        PYTHON = "C:\\Users\\speed\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"
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
                    "%PYTHON%" -m venv %VENV%
                    call %VENV%\\Scripts\\activate
                    "%PYTHON%" -m pip install --upgrade pip
                    "%PYTHON%" -m pip install ultralytics
                    if exist requirements.txt "%PYTHON%" -m pip install -r requirements.txt
                """
            }
        }

        stage('Run Model Test Script') {
            steps {
                bat """
                    call %VENV%\\Scripts\\activate
                    "%PYTHON%" modelTest.py
                """
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/*.txt', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: '**/*.json', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: '**/*.png', fingerprint: true, allowEmptyArchive: true
        }
    }

}
