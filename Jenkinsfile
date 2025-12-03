pipeline {
    agent any
    stages {
        stage('Run YOLO Model') {
    steps {
        sh """
            . venv/bin/activate
            python detect.py --weights best.pt --source val
        """
    }
}

        stage('Test') {
            steps {
                echo 'Jenkinsfile found and running!'
            }
        }
    }
}
