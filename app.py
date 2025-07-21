import gradio as gr
import requests
import json
import warnings
from typing import Optional, Dict, Any
import os
from fpdf import FPDF
import tempfile
from datetime import datetime
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Available models from Hugging Face API - Updated with working models
AVAILABLE_MODELS = {
    "Microsoft DialoGPT": {
        "model_name": "microsoft/DialoGPT-medium",
        "description": "Conversational AI model good for structured text"
    },
    "Google Flan-T5": {
        "model_name": "google/flan-t5-base",
        "description": "Text-to-text model excellent for professional writing"
    },
    "Salesforce BLIP": {
        "model_name": "Salesforce/blip-image-captioning-base",
        "description": "Multi-modal model (fallback to text generation)"
    },
    "GPT-2": {
        "model_name": "gpt2",
        "description": "Classic generative model for text completion"
    },
    "DistilBERT": {
        "model_name": "distilbert-base-uncased",
        "description": "Efficient transformer for text processing"
    }
}

# Alternative: Use text-generation models that are more likely to work
WORKING_MODELS = {
    "GPT-2": {
        "model_name": "gpt2",
        "description": "OpenAI's GPT-2 model for text generation"
    },
    "DistilGPT-2": {
        "model_name": "distilgpt2",
        "description": "Smaller, faster version of GPT-2"
    },
    "GPT-2 Large": {
        "model_name": "gpt2-large",
        "description": "Larger version of GPT-2 for better quality"
    },
    "T5-Small": {
        "model_name": "t5-small",
        "description": "Google's T5 model for text-to-text generation"
    }
}

class PDFGenerator:
    """Generate PDF documents from text content"""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def create_pdf(self, content: str, title: str, filename: str) -> str:
        """Create a PDF file from text content"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            
            # Title
            pdf.cell(0, 10, title, 0, 1, "C")
            pdf.ln(10)
            
            # Content
            pdf.set_font("Arial", size=11)
            
            # Split content into lines and add to PDF
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    # Handle long lines by wrapping
                    if len(line) > 90:
                        words = line.split(' ')
                        current_line = ""
                        for word in words:
                            if len(current_line + word + " ") <= 90:
                                current_line += word + " "
                            else:
                                if current_line:
                                    pdf.cell(0, 6, current_line.strip(), 0, 1)
                                current_line = word + " "
                        if current_line:
                            pdf.cell(0, 6, current_line.strip(), 0, 1)
                    else:
                        pdf.cell(0, 6, line, 0, 1)
                else:
                    pdf.ln(3)
            
            # Save PDF
            filepath = os.path.join(self.temp_dir, filename)
            pdf.output(filepath)
            return filepath
            
        except Exception as e:
            print(f"PDF generation error: {str(e)}")
            return None

class HuggingFaceAPI:
    """Handle Hugging Face API requests"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models/"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else None
        }
    
    def test_model_availability(self, model_name: str) -> bool:
        """Test if a model is available"""
        if not self.api_key:
            return False
        
        url = f"{self.base_url}{model_name}"
        
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json={"inputs": "test"},
                timeout=10
            )
            # Model is available if we get any response (even if it's loading)
            return response.status_code in [200, 503]
        except:
            return False
    
    def query_model(self, model_name: str, prompt: str, max_length: int = 512) -> str:
        """Query Hugging Face model via API"""
        if not self.api_key:
            return "Error: Hugging Face API key not found. Please set HUGGINGFACE_API_KEY environment variable."
        
        url = f"{self.base_url}{model_name}"
        
        # Different payload structures for different model types
        if "t5" in model_name.lower():
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
        else:
            # For GPT-2 and similar models
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": min(max_length, 250),  # Reduced for better API compatibility
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
        
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload,
                timeout=60  # Increased timeout
            )
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                print("Model is loading, waiting 20 seconds...")
                time.sleep(20)
                response = requests.post(
                    url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=60
                )
            
            if response.status_code == 404:
                return f"Error: Model '{model_name}' not found. Please try a different model."
            
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    generated = result[0]["generated_text"]
                    # Clean up the generated text
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    return generated
                else:
                    return str(result[0])
            elif isinstance(result, dict):
                if "generated_text" in result:
                    generated = result["generated_text"]
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    return generated
                else:
                    return str(result)
            else:
                return str(result)
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might be overloaded. Please try again later."
        except requests.exceptions.RequestException as e:
            if "404" in str(e):
                return f"Error: Model '{model_name}' is not available via the API. Please try a different model."
            return f"Error: API request failed - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

class ResumeBuilderApp:
    def __init__(self):
        self.hf_api = HuggingFaceAPI()
        self.pdf_generator = PDFGenerator()
        
    def build_resume_prompt(self, name, education, experience, skills, job_category, language, target_role=None):
        """Build optimized resume prompt"""
        target_section = f"\nTarget Role: {target_role}" if target_role else ""
        
        return f"""Write a professional {language} resume for:

Name: {name}
Education: {education}
Experience: {experience}
Skills: {skills}
Job Category: {job_category}{target_section}

Create a structured resume with these sections:

PROFESSIONAL SUMMARY:
"""

    def build_cover_letter_prompt(self, name, education, experience, skills, job_category, language, company_name=None):
        """Build optimized cover letter prompt"""
        company_section = f" for {company_name}" if company_name else ""
        
        return f"""Write a professional {language} cover letter{company_section}:

Applicant: {name}
Education: {education}
Experience: {experience}
Skills: {skills}
Job Category: {job_category}

Dear Hiring Manager,

"""

    def generate_resume(self, name, education, experience, skills, job_category, 
                       language, model_name, target_role, progress=gr.Progress()):
        """Generate resume"""
        
        if not all([name.strip(), education.strip(), experience.strip(), skills.strip(), job_category.strip()]):
            return "❌ Please fill in all required fields", None
        
        progress(0.2, desc="Building prompt...")
        
        prompt = self.build_resume_prompt(
            name, education, experience, skills, job_category, language, target_role
        )
        
        progress(0.5, desc="Generating resume with AI...")
        
        # Use WORKING_MODELS instead of AVAILABLE_MODELS
        model_info = WORKING_MODELS.get(model_name, WORKING_MODELS["GPT-2"])
        result = self.hf_api.query_model(model_info["model_name"], prompt, max_length=400)
        
        if result.startswith("Error"):
            progress(1.0, desc="Error occurred")
            return f"❌ {result}", None
        
        progress(0.8, desc="Creating PDF...")
        
        # Clean up the result and add structure
        clean_result = result.strip()
        if not clean_result:
            return "❌ Generated content is empty. Please try again with a different model.", None
        
        # Add basic structure to the result
        formatted_resume = f"""RESUME - {name.upper()}

PROFESSIONAL SUMMARY:
{clean_result}

EDUCATION:
{education}

SKILLS:
{skills}

WORK EXPERIENCE:
{experience}
"""
        
        # Generate PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"resume_{name.replace(' ', '_').replace('/', '_')}_{timestamp}.pdf"
        pdf_path = self.pdf_generator.create_pdf(formatted_resume, f"Resume - {name}", pdf_filename)
        
        progress(1.0, desc="Complete!")
        
        if pdf_path:
            return f"✅ **Resume Generated Successfully**\n\n{formatted_resume}", pdf_path
        else:
            return f"✅ **Resume Generated Successfully** (PDF creation failed)\n\n{formatted_resume}", None

    def generate_cover_letter(self, name, education, experience, skills, job_category,
                             language, model_name, company_name, progress=gr.Progress()):
        """Generate cover letter"""
        
        if not all([name.strip(), education.strip(), experience.strip(), skills.strip(), job_category.strip()]):
            return "❌ Please fill in all required fields", None
        
        progress(0.2, desc="Building prompt...")
        
        prompt = self.build_cover_letter_prompt(
            name, education, experience, skills, job_category, language, company_name
        )
        
        progress(0.5, desc="Generating cover letter with AI...")
        
        # Use WORKING_MODELS instead of AVAILABLE_MODELS
        model_info = WORKING_MODELS.get(model_name, WORKING_MODELS["GPT-2"])
        result = self.hf_api.query_model(model_info["model_name"], prompt, max_length=300)
        
        if result.startswith("Error"):
            progress(1.0, desc="Error occurred")
            return f"❌ {result}", None
        
        progress(0.8, desc="Creating PDF...")
        
        # Clean up the result and format as cover letter
        clean_result = result.strip()
        if not clean_result:
            return "❌ Generated content is empty. Please try again with a different model.", None
        
        company_text = company_name if company_name else "Your Company"
        
        formatted_letter = f"""COVER LETTER

{name}
Date: {datetime.now().strftime("%B %d, %Y")}

Dear Hiring Manager,

{clean_result}

Thank you for considering my application. I look forward to hearing from you.

Sincerely,
{name}
"""
        
        # Generate PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"cover_letter_{name.replace(' ', '_').replace('/', '_')}_{timestamp}.pdf"
        pdf_path = self.pdf_generator.create_pdf(formatted_letter, f"Cover Letter - {name}", pdf_filename)
        
        progress(1.0, desc="Complete!")
        
        if pdf_path:
            return f"✅ **Cover Letter Generated Successfully**\n\n{formatted_letter}", pdf_path
        else:
            return f"✅ **Cover Letter Generated Successfully** (PDF creation failed)\n\n{formatted_letter}", None

    def create_interface(self):
        """Create Gradio interface"""
        
        custom_css = """
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
        }
        .section-header {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            margin: 1rem 0;
        }
        .info-box {
            background: #f0f8ff;
            border: 1px solid #cce7ff;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        """
        
        with gr.Blocks(css=custom_css, title="AI Resume & Cover Letter Builder", theme=gr.themes.Soft()) as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🚀 AI Resume & Cover Letter Builder</h1>
                <p>Create professional resumes and cover letters with AI models</p>
            </div>
            """)
            
            # API Key info
            gr.HTML("""
            <div class="info-box">
                <strong>📝 Setup Required:</strong> Set your Hugging Face API key as environment variable 
                <code>HUGGINGFACE_API_KEY</code> to use this application.
                <br>Get your free API key at: <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a>
            </div>
            """)
            
            # Warning about models
            gr.HTML("""
            <div class="warning-box">
                <strong>⚠️ Note:</strong> This version uses basic GPT-2 models which are more reliable but may produce shorter text. 
                For better results, consider using local models or paid APIs like OpenAI.
            </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">🤖 AI Model Selection</div>')
                    model_dropdown = gr.Dropdown(
                        choices=list(WORKING_MODELS.keys()),
                        value="GPT-2",
                        label="AI Model",
                        info="Select the AI model for document generation"
                    )
                    
                    # Display model info
                    def update_model_info(model_name):
                        return WORKING_MODELS[model_name]["description"]
                    
                    model_info = gr.Textbox(
                        label="Model Description",
                        value=WORKING_MODELS["GPT-2"]["description"],
                        interactive=False
                    )
                    
                    model_dropdown.change(
                        fn=update_model_info,
                        inputs=[model_dropdown],
                        outputs=[model_info]
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">📝 Personal Information</div>')
                    
                    name_input = gr.Textbox(
                        label="Full Name *",
                        placeholder="Enter your full name",
                        max_lines=1
                    )
                    
                    language_input = gr.Dropdown(
                        choices=["English", "Spanish", "French", "German", "Italian", "Portuguese"],
                        value="English",
                        label="Language"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">🎓 Professional Details</div>')
                    
                    education_input = gr.Textbox(
                        label="Education *",
                        placeholder="e.g., Bachelor's in Computer Science, XYZ University, 2020",
                        lines=2
                    )
                    
                    experience_input = gr.Textbox(
                        label="Work Experience *",
                        placeholder="e.g., 3 years as Software Developer at ABC Corp, led team of 5 developers...",
                        lines=4
                    )
                    
                    skills_input = gr.Textbox(
                        label="Skills *",
                        placeholder="e.g., Python, JavaScript, Project Management, Communication",
                        lines=2
                    )
                    
                    job_category_input = gr.Textbox(
                        label="Job Category/Industry *",
                        placeholder="e.g., Software Development, Marketing, Healthcare",
                        max_lines=1
                    )
            
            with gr.Accordion("🎯 Additional Options", open=False):
                target_role_input = gr.Textbox(
                    label="Target Role (Optional)",
                    placeholder="e.g., Senior Software Engineer, Marketing Manager"
                )
                
                company_name_input = gr.Textbox(
                    label="Company Name (for Cover Letter)",
                    placeholder="e.g., Google, Microsoft, Local Hospital"
                )
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">📄 Resume Generator</div>')
                    resume_btn = gr.Button("🔥 Generate Resume", variant="primary", size="lg")
                    resume_output = gr.Textbox(
                        label="Generated Resume", 
                        lines=15, 
                        show_copy_button=True,
                        placeholder="Your generated resume will appear here..."
                    )
                    resume_download = gr.File(label="📥 Download Resume PDF")
                    
                with gr.Column():
                    gr.HTML('<div class="section-header">📧 Cover Letter Generator</div>')
                    cover_letter_btn = gr.Button("✉️ Generate Cover Letter", variant="primary", size="lg")
                    cover_letter_output = gr.Textbox(
                        label="Generated Cover Letter", 
                        lines=15, 
                        show_copy_button=True,
                        placeholder="Your generated cover letter will appear here..."
                    )
                    cover_letter_download = gr.File(label="📥 Download Cover Letter PDF")
            
            # Examples section
            gr.HTML('<div class="section-header">💡 Example</div>')
            gr.Examples(
                examples=[
                    [
                        "Sarah Johnson", 
                        "Bachelor's in Marketing, State University, 2020; Google Analytics Certified", 
                        "3 years as Digital Marketing Specialist at TechCorp. Managed social media campaigns reaching 100K+ users, increased engagement by 45%, led content strategy for product launches",
                        "Digital Marketing, SEO/SEM, Content Creation, Google Analytics, Social Media Management, Project Management, Adobe Creative Suite",
                        "Digital Marketing", 
                        "English", 
                        "GPT-2", 
                        "Senior Marketing Manager"
                    ]
                ],
                inputs=[
                    name_input, education_input, experience_input, skills_input,
                    job_category_input, language_input, model_dropdown, target_role_input
                ],
                label="Click to load example data"
            )
            
            # Event handlers
            resume_btn.click(
                fn=self.generate_resume,
                inputs=[
                    name_input, education_input, experience_input, skills_input,
                    job_category_input, language_input, model_dropdown, target_role_input
                ],
                outputs=[resume_output, resume_download],
                show_progress=True
            )
            
            cover_letter_btn.click(
                fn=self.generate_cover_letter,
                inputs=[
                    name_input, education_input, experience_input, skills_input,
                    job_category_input, language_input, model_dropdown, company_name_input
                ],
                outputs=[cover_letter_output, cover_letter_download],
                show_progress=True
            )
        
        return interface

def main():
    """Main function"""
    app = ResumeBuilderApp()
    interface = app.create_interface()
    
    # Launch configuration
    launch_kwargs = {
        "show_error": True,
        "share": False,
        "server_name": "0.0.0.0",
        "server_port": 7860 if os.getenv("SPACE_ID") else None
    }
    
    interface.launch(**launch_kwargs)

if __name__ == "__main__":
    main()