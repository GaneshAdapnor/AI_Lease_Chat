from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class RenewalOption(BaseModel):
    number_of_options: Optional[int] = Field(description="Number of renewal options available")
    term_years: Optional[int] = Field(description="Term in years for each renewal")
    notice_period: Optional[str] = Field(description="Notice period required (e.g., earliest/latest notice)")
    notes: Optional[str] = Field(description="Additional notes on renewal options")

class TerminationClause(BaseModel):
    description: Optional[str] = Field(description="Description of early termination rights")
    sales_kickout: Optional[str] = Field(description="Sales kickout details")
    co_tenancy: Optional[str] = Field(description="Co-tenancy details")

class LeaseSummary(BaseModel):
    tenant: Optional[str] = Field(description="Name of the Tenant")
    landlord: Optional[str] = Field(description="Name of the Landlord")
    dba_name: Optional[str] = Field(description="Doing Business As (DBA) name")
    address: Optional[str] = Field(description="Leased premises address")
    leased_area_sqft: Optional[str] = Field(description="Leased area in square feet")
    lease_start_date: Optional[str] = Field(description="Lease Commencement Date or Start Date")
    lease_end_date: Optional[str] = Field(description="Lease Expiration Date or End Date")
    rent_amount: Optional[str] = Field(description="Rent Amount details (Annual, Monthly, PSF)")
    security_deposit: Optional[str] = Field(description="Security deposit amount")
    renewal_options: Optional[List[RenewalOption]] = Field(description="Details of renewal options")
    termination_clauses: Optional[List[TerminationClause]] = Field(description="Details of early termination clauses")
    special_provisions: Optional[str] = Field(description="Any special compliance provisions or unique notes")
    permitted_use: Optional[str] = Field(description="Permitted Use / Exclusive Use clauses")

def extract_lease_summary(text: str) -> LeaseSummary:
    """
    Uses gpt-4o-mini with structured output to parse the provided text into the LeaseSummary schema.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(LeaseSummary)
    
    prompt = (
        "You are an expert paralegal. Extract the requested lease summary information from the following lease document text.\n"
        "If a field is not present or ambiguous, return null or an empty string, but extract as much as you can accurately.\n\n"
        f"Document Text:\n{text[:100000]}"
    )
    
    try:
        result = structured_llm.invoke(prompt)
        return result
    except Exception as e:
        print(f"Extraction error: {e}")
        return None
