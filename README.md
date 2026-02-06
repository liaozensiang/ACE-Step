# This is a fork of https://github.com/ace-step/ACE-Step for dgx spark (nvidia GB10)
## What I've changed:
* replace torchaudio with soundfile & librosa
* changed Dockerfile for easier usage
## How to run on DGX Spark?
1. run `git clone https://github.com/liaozensiang/ACE-Step.git && cd ACE-Step `
2. run `docker compose build`
3. run `docker compose up`
4. Go to http://localhost:7865 to use UI

## More instructions please go to the origin repo https://github.com/ace-step/ACE-Step/blob/main/README.md
