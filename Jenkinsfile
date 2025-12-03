pipeline {
    agent any
    stages {
        stage('Run YOLO Model') {
    steps {
        sh """
            . venv/bin/activate
            python detect.py --weights yoloweights.pt --source data/test_images
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
