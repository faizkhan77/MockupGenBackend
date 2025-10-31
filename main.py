import os
import asyncio
import base64
import mimetypes
import time
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize the client once globally
client = genai.Client(api_key=api_key)

# ✅ SOLUTION: Create a semaphore to limit concurrent requests to the API.
# This prevents the "model is overloaded" error by ensuring we don't send
# all 6 requests at the exact same time. A value of 3 is a safe starting point.
SEMAPHORE = asyncio.Semaphore(3)

# --- FastAPI App Initialization ---
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class PromptRequest(BaseModel):
    prompt: str

class EnhancedPromptResponse(BaseModel):
    enhanced_prompt: str

class ImageResponse(BaseModel):
    images: list[str]

# --- Helper Function ---
async def generate_single_image(image_bytes, mime_type, master_prompt, index):
    """Generates a single image, respecting the concurrency limit set by the semaphore."""
    # ✅ SOLUTION: The 'async with SEMAPHORE:' block ensures that no more than
    # 3 of these functions will run their core logic at the same time.
    async with SEMAPHORE:
        print(f"Starting generation for image {index}...")
        try:
            contents = [
                types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime_type)),
                types.Part.from_text(text=master_prompt)
            ]
            config = types.GenerateContentConfig(response_modalities=["IMAGE"])
            
            stream = client.models.generate_content_stream(
                model="gemini-2.5-flash-image-preview",
                contents=contents,
                config=config
            )

            for chunk in stream:
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                
                for part in chunk.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        img_data = part.inline_data.data
                        b64 = base64.b64encode(img_data).decode("utf-8")

                        os.makedirs("images", exist_ok=True)
                        timestamp = int(time.time())
                        ext = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                        file_name = f"images/mockup_{timestamp}_{index}{ext}"
                        with open(file_name, "wb") as f:
                            f.write(img_data)

                        print(f"Finished generation for image {index}.")
                        return b64
        except Exception as e:
            print(f"Error generating image {index}:", e)
            return None


@app.post("/api/generate-mockups")
async def generate_mockups(
    prompt: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        print("Received request to generate 6 images with controlled concurrency...")
        image_bytes = await image.read()
        mime_type = mimetypes.guess_type(image.filename)[0]

        master_prompt = f"""
            You are a professional mockup designer.
            Generate a realistic, high-quality mockup of the below item that naturally and attractively integrates the provided logo or design.
            Further user specifications: {prompt}
            Make sure the output looks professional, photo-realistic, and studio-quality.
        """
        
        # This creates all 6 tasks, but the semaphore inside generate_single_image
        # will control their execution.
        tasks = [generate_single_image(image_bytes, mime_type, master_prompt, i) for i in range(6)]
        results = await asyncio.gather(*tasks)

        images_b64 = [r for r in results if r]
        print(f"Successfully generated {len(images_b64)} images.")

        if not images_b64:
            raise HTTPException(status_code=500, detail="No image data returned from Gemini API")

        return {"images": images_b64, "count": len(images_b64)}

    except Exception as e:
        print(f"Error in generate_mockups endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def enhance_prompt_with_gemini(user_prompt: str) -> str:
    """Calls Gemini to rewrite and enhance a user's prompt."""
    master_prompt = f"""
    You are a creative assistant for an AI mockup generator. 
    Your task is to rewrite the user's input in a better way,  
    for the best image generation results. Keep the core subject but go upto max 1 lines only
    Return ONLY the enhanced prompt.
    User's prompt: "{user_prompt}"
    Enhanced prompt:
    """
    try:
        def generate():
            return client.models.generate_content(
                model="models/gemini-flash-latest",
                contents=[master_prompt]
            )
        response = await asyncio.to_thread(generate)
        return response.text.strip()
    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        return user_prompt

@app.post("/api/enhance-prompt", response_model=EnhancedPromptResponse)
async def enhance_prompt(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    enhanced_prompt = await enhance_prompt_with_gemini(request.prompt)
    
    return EnhancedPromptResponse(enhanced_prompt=enhanced_prompt)