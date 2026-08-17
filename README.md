# [EEG FM Digest](https://iroblesrazzaq.github.io/EEG-FM-Digest/)

I built this digest to track the influx of EEG FM papers! There are a lot of EEG FM preprints on arXiv that are hard to find, and I wanted an easy way of tracking them so I built this digest to track and classify EEG FM papers. You can filter for paper types, architectures, etc through search. Via github actions, it runs a daily arxiv query, llm filtering, then llm summary with gemma 4 31b for any accepted papers. I've found it useful, and I hope other people find it useful. It also updates the [awesome-eeg-fm](https://github.com/iroblesrazzaq/awesome-eeg-fm) repo automatically. 

Will build a feature where other people can submit an arxiv link that automatically triggers the llm triage, and if it passes, auto do summary and update. If you submit a paper and don't see it, it must've failed, just let me know by creating an issue and I'll add it myself.
