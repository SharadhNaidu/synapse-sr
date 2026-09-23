# Offline and on-premises use

synapse-sr never needs the network for processing. The only network access is the one-time download of the
weights by `Pro.from_pretrained()`, and that is skipped when you pass a local file.

## Air-gapped machines

1. On a connected machine, download the checkpoint from
   [Hugging Face](https://huggingface.co/SharadhNaiduTrains/synapse-sr) and build a wheelhouse:

    ```bash
    pip download synapse-sr -d wheelhouse
    ```

2. Copy `wheelhouse/` and the `.safetensors` file to the target machine, then:

    ```bash
    pip install --no-index --find-links wheelhouse synapse-sr
    synapse-sr scene.tif scene_2m.tif --weights /opt/models/synapse-pro-v1.safetensors
    ```

```python
from synapse_sr import Pro
model = Pro.from_pretrained(weights="/opt/models/synapse-pro-v1.safetensors")
```

## Cache

Downloaded weights are stored in `~/.cache/synapse`, or in `$SYNAPSE_CACHE` if set. They are verified by
SHA-256 on every load. A corrupted or tampered file raises an error that names the expected and actual hashes.

## Your own checkpoints

```python
model.save_pretrained("my-model.safetensors")         # model + sensor operator in one file
Pro.from_pretrained(weights="my-model.safetensors")
```
