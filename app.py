import gradio as gr
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    GenerationConfig
)
import warnings
from typing import Optional, Dict, Any
import gc
import os
from fpdf import FPDF
import tempfile
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Lightweight models optimized for deployment
AVAILABLE_MODELS = {
    "GPT-2 Small": {
        "model_name": "gpt2",
        "model_type": "causal",
        "max_length": 1024,
        "description": "Fast and efficient for quick generation"
    },
    "DistilGPT-2": {
        "model_name": "distilgpt2", 
        "model_type": "causal",
        "max_length": 1024,
        "description": "Optimized lightweight model"
    },
    "FLAN-T5 Small": {
        "model_name": "google/flan-t5-small",
        "model_type": "seq2seq",
        "max_length": 512,
        "description": "Instruction-tuned model for structured tasks"
    },
    "T5 Small": {
        "model_name": "t5-small",
        "model_type": "seq2seq", 
        "max_length": 512,
        "description": "Compact T5 model"
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
                    if len(line) > 80:
                        words = line.split(' ')
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 80:
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

class ModelManager:
    """Manages model loading with memory optimization for deployment"""
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_name: str, model_info: Dict[str, Any]):
        """Load model with deployment optimizations"""
        if self.current_model_name == model_name and self.current_model is not None:
            return self.current_model, self.current_tokenizer
        
        self.clear_model()
        
        try:
            # Load tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_info["model_name"],
                trust_remote_code=True
            )
            
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            
            # Load model with memory optimization
            if model_info["model_type"] == "causal":
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_info["model_name"],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                self.current_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_info["model_name"],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            self.current_model = self.current_model.to(self.device)
            self.current_model_name = model_name
            
            return self.current_model, self.current_tokenizer
            
        except Exception as e:
            print(f"Model loading error: {str(e)}")
            return None, None
    
    def clear_model(self):
        """Clear model from memory"""
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def generate_text(self, prompt: str, model_name: str, max_new_tokens: int = 400):
        """Generate text with optimized parameters"""
        model_info = AVAILABLE_MODELS.get(model_name)
        if not model_info:
            return "Model not available"
        
        model, tokenizer = self.load_model(model_name, model_info)
        if model is None or tokenizer is None:
            return f"Could not load model {model_name}"
        
        try:
            inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, 
                                    max_length=model_info["max_length"]//2)
            inputs = inputs.to(self.device)
            
            generation_config = GenerationConfig(
                max_new_tokens=min(max_new_tokens, model_info["max_length"]//2),
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            with torch.no_grad():
                if model_info["model_type"] == "causal":
                    outputs = model.generate(inputs, generation_config=generation_config)
                    generated_ids = outputs[0][inputs.shape[1]:]
                else:
                    outputs = model.generate(inputs, generation_config=generation_config)
                    generated_ids = outputs[0]
                
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                return generated_text.strip()
                
        except Exception as e:
            return f"Generation error: {str(e)}"

class ResumeBuilderApp:
    def __init__(self):
        self.model_manager = ModelManager()
        self.pdf_generator = PDFGenerator()
        
    def build_resume_prompt(self, name, education, experience, skills, job_category, language, target_role=None):
        """Build optimized resume prompt"""
        target_section = f"Target Role: {target_role}\n" if target_role else ""
        
        return f"""Create a professional resume in {language}:

Name: {name}
Education: {education}
Experience: {experience}
Skills: {skills}
Job Category: {job_category}
{target_section}

Generate a well-structured resume with:
- Professional header
- Summary (2-3 sentences)
- Skills section
- Experience with achievements
- Education section

Resume:
"""

    def build_cover_letter_prompt(self, name, education, experience, skills, job_category, language, company_name=None):
        """Build optimized cover letter prompt"""
        company_section = f"Company: {company_name}\n" if company_name else ""
        
        return f"""Write a professional cover letter in {language}:

Name: {name}
Education: {education}
Experience: {experience}
Skills: {skills}
Job Category: {job_category}
{company_section}

Create a compelling cover letter with professional opening, relevant experience, and strong closing.

Cover Letter:
"""

    def generate_resume(self, name, education, experience, skills, job_category, 
                       language, model_name, target_role, progress=gr.Progress()):
        """Generate resume"""
        
        if not all([name, education, experience, skills, job_category]):
            return "Please fill in all required fields", None, None
        
        progress(0.2, desc="Building prompt...")
        
        prompt = self.build_resume_prompt(
            name, education, experience, skills, job_category, language, target_role
        )
        
        progress(0.5, desc="Generating resume...")
        
        result = self.model_manager.generate_text(prompt, model_name, max_new_tokens=500)
        
        if result.startswith("Error") or result.startswith("Could not") or result.startswith("Model not"):
            progress(1.0, desc="Error occurred")
            return f"❌ {result}", None, None
        
        progress(0.8, desc="Creating PDF...")
        
        # Generate PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"resume_{name.replace(' ', '_')}_{timestamp}.pdf"
        pdf_path = self.pdf_generator.create_pdf(result, f"Resume - {name}", pdf_filename)
        
        progress(1.0, desc="Complete!")
        
        return f"✅ **Resume Generated Successfully**\n\n{result}", pdf_path, pdf_path

    def generate_cover_letter(self, name, education, experience, skills, job_category,
                             language, model_name, company_name, progress=gr.Progress()):
        """Generate cover letter"""
        
        if not all([name, education, experience, skills, job_category]):
            return "Please fill in all required fields", None, None
        
        progress(0.2, desc="Building prompt...")
        
        prompt = self.build_cover_letter_prompt(
            name, education, experience, skills, job_category, language, company_name
        )
        
        progress(0.5, desc="Generating cover letter...")
        
        result = self.model_manager.generate_text(prompt, model_name, max_new_tokens=400)
        
        if result.startswith("Error") or result.startswith("Could not") or result.startswith("Model not"):
            progress(1.0, desc="Error occurred")
            return f"❌ {result}", None, None
        
        progress(0.8, desc="Creating PDF...")
        
        # Generate PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"cover_letter_{name.replace(' ', '_')}_{timestamp}.pdf"
        pdf_path = self.pdf_generator.create_pdf(result, f"Cover Letter - {name}", pdf_filename)
        
        progress(1.0, desc="Complete!")
        
        return f"✅ **Cover Letter Generated Successfully**\n\n{result}", pdf_path, pdf_path

    def create_interface(self):
        """Create Gradio interface optimized for deployment"""
        
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
        """
        
        with gr.Blocks(css=custom_css, title="AI Resume & Cover Letter Builder") as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🚀 AI Resume & Cover Letter Builder</h1>
                <p>Create professional resumes and cover letters with AI assistance</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">🤖 Model Selection</div>')
                    model_dropdown = gr.Dropdown(
                        choices=list(AVAILABLE_MODELS.keys()),
                        value="DistilGPT-2",
                        label="AI Model",
                        info="Select model for document generation"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">📝 Personal Information</div>')
                    
                    name_input = gr.Textbox(
                        label="Full Name *",
                        placeholder="Enter your full name"
                    )
                    
                    language_input = gr.Dropdown(
                        choices=["English", "Spanish", "French", "German"],
                        value="English",
                        label="Language"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">🎓 Professional Details</div>')
                    
                    education_input = gr.Textbox(
                        label="Education *",
                        placeholder="Your educational background",
                        lines=2
                    )
                    
                    experience_input = gr.Textbox(
                        label="Work Experience *",
                        placeholder="Your work experience and achievements",
                        lines=3
                    )
                    
                    skills_input = gr.Textbox(
                        label="Skills *",
                        placeholder="Your technical and professional skills",
                        lines=2
                    )
                    
                    job_category_input = gr.Textbox(
                        label="Job Category *",
                        placeholder="Target job field or industry"
                    )
            
            with gr.Accordion("Additional Options", open=False):
                target_role_input = gr.Textbox(
                    label="Target Role",
                    placeholder="Specific position you're targeting"
                )
                
                company_name_input = gr.Textbox(
                    label="Company Name (for Cover Letter)",
                    placeholder="Target company name"
                )
            
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="section-header">📄 Resume</div>')
                    resume_btn = gr.Button("Generate Resume", variant="primary", size="lg")
                    resume_output = gr.Textbox(label="Generated Resume", lines=15, show_copy_button=True)
                    resume_download = gr.File(label="Download Resume PDF", visible=False)
                    
                with gr.Column():
                    gr.HTML('<div class="section-header">📧 Cover Letter</div>')
                    cover_letter_btn = gr.Button("Generate Cover Letter", variant="primary", size="lg")
                    cover_letter_output = gr.Textbox(label="Generated Cover Letter", lines=15, show_copy_button=True)
                    cover_letter_download = gr.File(label="Download Cover Letter PDF", visible=False)
            
            # Examples
            gr.Examples(
                examples=[
                    [
                        "Alex Johnson", "Bachelor's in Marketing, State University, 2020",
                        "3 years as Marketing Coordinator at TechCorp, managed social media campaigns",
                        "Digital Marketing, SEO, Content Creation, Analytics",
                        "Marketing", "English", "DistilGPT-2", "Marketing Manager"
                    ]
                ],
                inputs=[
                    name_input, education_input, experience_input, skills_input,
                    job_category_input, language_input, model_dropdown, target_role_input
                ],
                label="Example Profile"
            )
            
            # Event handlers
            resume_btn.click(
                fn=self.generate_resume,
                inputs=[
                    name_input, education_input, experience_input, skills_input,
                    job_category_input, language_input, model_dropdown, target_role_input
                ],
                outputs=[resume_output, resume_download, gr.File(visible=True)],
                show_progress=True
            )
            
            cover_letter_btn.click(
                fn=self.generate_cover_letter,
                inputs=[
                    name_input, education_input, experience_input, skills_input,
                    job_category_input, language_input, model_dropdown, company_name_input
                ],
                outputs=[cover_letter_output, cover_letter_download, gr.File(visible=True)],
                show_progress=True
            )
        
        return interface

def main():
    """Main function for deployment"""
    app = ResumeBuilderApp()
    interface = app.create_interface()
    
    # Launch configuration for different platforms
    launch_kwargs = {
        "show_error": True,
        "quiet": True,
    }
    
    # Detect deployment environment
    if os.getenv("SPACE_ID"):  # Hugging Face Spaces
        launch_kwargs.update({
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False
        })
    else:  # Local or Colab
        launch_kwargs.update({
            "share": True,
            "debug": False
        })
    
    interface.launch(**launch_kwargs)

if __name__ == "__main__":
    main()