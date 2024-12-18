# Known Issues

- [ ] Missing "Training Paused" notification on tab switch. When switching to a previous sub-tab of "Training an Agent", while training is happening, there should be a "Training Paused" message, but none is displayed.
- [ ] Training not paused when switching away from "Training an Agent" tab. This only happens within the sub-tabs of "Training an Agent"

# Features

- [ ] Reset Button for Dataset Input and Simulation Settings sub-tabs. Would reset all components to their default values.
- [ ] Automatically switch tabs on successful submission. Gradio seems to be missing a way to switch tabs programmatically, but maybe we can do more checking of inputs on the fly and just make the next tab interactive when all required values are filled. Then we can do whatever computation necessary (e.g. kbest) on tab selection event. This would require significant changes to the current structure, removing the original submission button and processing status, maybe change the kbest functionality to it can be cancelled, etc.
- [ ] Model implementation: separate the config option for shipment logs per episode, and final simulation output. Then add checkbox in UI to control shipment logs per episode.
- [ ] Better error messages. If something goes wrong because of some input, it would be nice the highlight the input and put the message next to it...
- [ ] UI Manual.