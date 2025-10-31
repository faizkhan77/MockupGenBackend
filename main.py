import os
import asyncio
import base64
import mimetypes
import time
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.dialects.mysql import LONGTEXT
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import google.genai as genai
from google.genai import types
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlmodel import SQLModel, Session, create_engine, Field, Relationship, select
import sqlalchemy

# Load environment variables from .env file
load_dotenv()

# --- Security and Authentication Setup ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

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


def get_session():
    with Session(engine) as session:
        yield session

# --- Database Models (re-defined here for simplicity, could be in a separate file) ---
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str
    images: List["MockupImage"] = Relationship(back_populates="user")

class MockupImage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    base64_image: str = Field(sa_column=sqlalchemy.Column(LONGTEXT))
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    user: Optional[User] = Relationship(back_populates="images")

# --- Pydantic Models for API ---
class UserCreate(BaseModel):
    username: str
    password: str

class UserPublic(BaseModel):
    id: int
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str

class MockupImagePublic(BaseModel):
    id: int
    prompt: str
    base64_image: str

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
    async with SEMAPHORE:
        try:
            contents = [
                types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime_type)),
                types.Part.from_text(text=master_prompt)
            ]
            config = types.GenerateContentConfig(response_modalities=["IMAGE"])
            
            # ✅ FIXED: Changed the function call back to using the 'model=' keyword argument,
            # which is the correct syntax for the SDK. This resolves the TypeError.
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
                        # Return just the base64 string for saving and sending to the user
                        return base64.b64encode(part.inline_data.data).decode("utf-8")
        except Exception as e:
            print(f"Error generating image {index}:", e)
            return None



# --- Authentication Helper Functions ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a password, respecting the 72-byte limit of bcrypt."""
    password_bytes = plain_password.encode('utf-8')
    # Truncate the user's input password to 72 bytes before verification
    return pwd_context.verify(password_bytes[:72], hashed_password)

def get_password_hash(password: str) -> str:
    print("Password type:", type(password), "Length:", len(password))
    password_bytes = password.encode('utf-8') if isinstance(password, str) else password
    return pwd_context.hash(password_bytes[:72])


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = session.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise credentials_exception
    return user

# --- API Endpoints ---

@app.post("/api/register", response_model=UserPublic)
def register_user(user_create: UserCreate, session: Session = Depends(get_session)):
    db_user = session.exec(select(User).where(User.username == user_create.username)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user_create.password)
    new_user = User(username=user_create.username, hashed_password=hashed_password)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user

@app.post("/api/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/users/me", response_model=UserPublic)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/api/generate-mockups")
async def generate_mockups(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    try:
        image_bytes = await image.read()
        mime_type = mimetypes.guess_type(image.filename)[0]
        
        master_prompt = f"""
            You are a professional mockup designer.
            Generate a realistic, high-quality mockup of the below item that naturally and attractively integrates the provided logo or design.
            Further user specifications: {prompt}
            Make sure the output looks professional, photo-realistic, and studio-quality.
        """
        
        tasks = [generate_single_image(image_bytes, mime_type, master_prompt, i) for i in range(6)]
        results = await asyncio.gather(*tasks)

        images_b64 = [r for r in results if r]
        
        # Save successful generations to the database for the logged-in user
        if images_b64:
            print(f"Successfully generated {len(images_b64)} images. Saving to database...")
            for b64_img in images_b64:
                db_mockup = MockupImage(prompt=prompt, base64_image=b64_img, user_id=current_user.id)
                session.add(db_mockup)
            session.commit()
            print("Save complete.")
        else:
            print("Generation failed to produce images.")


        if not images_b64:
            raise HTTPException(status_code=500, detail="No image data returned from Gemini API")
            
        return {"images": images_b64, "count": len(images_b64)}

    except Exception as e:
        print(f"Error in generate_mockups endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/my-mockups", response_model=List[MockupImagePublic])
async def get_my_mockups(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    # The relationship 'images' on the User object is automatically populated by SQLModel/SQLAlchemy
    return current_user.images


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