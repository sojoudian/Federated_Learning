# Federated_Learning
My federated learning practices:

`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python  # Python 3
sudo pip3 install --upgrade virtualenv  # system-wide install`

`virtualenv --python python3 "venv"
source "venv/bin/activate"
pip install --upgrade pip`

`pip install --upgrade tensorflow_federated`

`python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"
`
