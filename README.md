1. Fork this project
2. Create an environment and give it any name you like.

<p align="center">
  <img src="Pasted_image_20251021001024.png" alt="Environment Secrets" width="1000"/>
</p>
  
  3. Add a secret named CIVITAI_API_KEY. You can obtain the API value from the following link. (https://education.civitai.com/civitais-guide-to-downloading-via-api/).

<p align="center">
  <img src="Pasted_image_20251021001430.png" alt="Add API Key" width="1000"/>
</p>

4. Modify the prompt in prepare_data.py if needed. If prepare_data.py exists in the root directory, the CI tool will overwrite the downloaded file; otherwise, the default value from the server will be used.
5. Check modelVersionId and run GitHub Action.

<p align="center">
  <img src="Pasted_image_20251021001559.png" alt="Model Link" width="1000"/>
</p>

