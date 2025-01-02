from builtins import print
import requests

def callback_to_nextjs(job_id, result,vercelUrl):

    # Get the dynamic Vercel URL
    nextjs_url = vercelUrl

    # Construct the webhook URL
    webhook_url = f"{nextjs_url}/api/NER-model/webhook"

    data = {
        "job_id": job_id,
        "result": result
    }

    # Send the result to Next.js via the webhook
    try:
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending webhook to Next.js: {e}")
