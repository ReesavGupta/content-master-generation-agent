import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, START, END
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('content_master.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


######################## Define State and Models ########################

class ContentMasterState(BaseModel):
    user_request: str = ""
    research_query: str = ""
    search_results: List[Dict] = []
    verified_sources: List[Dict] = []
    content_outline: Dict = {}
    generated_content: Dict = {}
    visual_elements: List[Dict] = []
    final_output: str = ""
    content_type: str = ""  # "presentation", "document", "webpage"


class CredibilityScore(BaseModel):
    score: float = Field(description="Credibility score from 0 to 10")


class VisualElement(BaseModel):
    type: str = Field(description="Type of visual element, e.g. 'bar_chart', 'line_chart', etc.")
    data_points: List[Any] = Field(description="Key data points to visualize")


class GeneratedSection(BaseModel):
    text: str = Field(description="Generated content for the section")
    sources: List[str] = Field(description="List of source URLs")


class TemplateConfig(BaseModel):
    layout: Dict[str, Any]
    colors: Dict[str, Any]
    typography: Dict[str, Any]
    visual_hierarchy: Dict[str, Any]


######################## Create tools and models ########################

llm = ChatGroq(model="llama3-8b-8192")  # Make sure your API key is set properly for this model

search_tool = TavilySearchResults(max_results=5, include_answer=True, include_raw_content=True, include_images=True)


def get_search_tool():
    return search_tool


def query_analyzer(state: ContentMasterState):
    # Determine content type and refine search query
    response = llm.invoke(f"""
    Analyze the user request: {state.user_request}
    Determine:
    1. Content Type (presentation/document/webpage)
    2. Refined Search Query

    Output Format:
    {{
        "content_type": "...",
        "research_query": "..."
    }}
    Respond ONLY with a valid JSON object, no extra text.
    """)
    try:
        parsed_response = json.loads(response.content)
    except Exception as e:
        logger.error(f"Failed to parse LLM response as JSON in query_analyzer: {response.content}")
        raise
    return {
        "content_type": parsed_response["content_type"],
        "research_query": parsed_response["research_query"]
    }


def research_agent(state: ContentMasterState):
    # Perform web search
    search_results = search_tool.invoke(state.research_query)
    return {
        "search_results": search_results
    }


def source_verifier(state: ContentMasterState):
    # Verify and rank sources
    structured_llm = llm.with_structured_output(CredibilityScore, method="json_mode")
    verified_sources = []
    for result in state.search_results:
        credibility_check = structured_llm.invoke(f"""
        Evaluate the credibility of this source:
        Title: {result['title']}
        Link: {result['url']}
        
        Criteria:
        - Relevance to research query
        - Source authority
        - Recency of information
        
        Return your answer as a JSON object with a single field 'score', e.g.:
        {{"score": 8.5}}
        """)
        score = credibility_check.score
        if score > 7:
            verified_sources.append(result)
    return {
        "verified_sources": verified_sources
    }


def content_planner(state: ContentMasterState):
    # Create structured outline based on content type
    outline_prompt = f"""
    Create a structured outline for a {state.content_type} 
    on {state.research_query} using verified sources.
    
    Sources: {[src['title'] for src in state.verified_sources]}
    
    Output a JSON outline with sections and key points.
    Respond ONLY with a valid JSON object, no extra text.
    """
    
    try:
        outline = llm.invoke(outline_prompt)
        outline_json = json.loads(outline.content)
        return {
            "content_outline": outline_json
        }
    except Exception as e:
        logger.error(f"Error in content planning or parsing: {outline.content}")
        return {
            "content_outline": {
                "Introduction": "Unable to generate outline",
                "Main Content": "Error occurred during planning"
            }
        }


def content_generator(state: ContentMasterState):
    # Generate content for each section
    structured_llm = llm.with_structured_output(GeneratedSection, method="json_mode")
    generated_content = {}
    for section, details in state.content_outline.items():
        try:
            content = structured_llm.invoke(f"""
            Generate content for section: {section}
            Context: {details}
            Sources: {[src['url'] for src in state.verified_sources]}
            
            Return your answer as a JSON object with fields 'text' and 'sources', e.g.:
            {{"text": "...", "sources": ["url1", "url2"]}}
            """)
            generated_content[section] = {
                "text": content.text,
                "sources": content.sources
            }
        except Exception as e:
            logger.error(f"Failed to parse LLM response as JSON in content_generator for section '{section}': {getattr(content, 'content', str(content))}")
            generated_content[section] = {
                "text": getattr(content, 'content', str(content)),
                "sources": [src['url'] for src in state.verified_sources]
            }
    return {
        "generated_content": generated_content
    }


def visual_creator(state: ContentMasterState):
    # Generate visual elements based on content
    structured_llm = llm.with_structured_output(VisualElement, method="json_mode")
    visual_elements = []
    
    for section, content in state.generated_content.items():
        try:
            visual_suggestion = structured_llm.invoke(f"""
            Suggest a visual element for section: {section}
            Content summary: {content['text'][:200]}
            
            Return your answer as a JSON object with fields 'type' and 'data_points', e.g.:
            {{"type": "bar_chart", "data_points": ["Year", "Missions", "Budget"]}}
            """)
            visual_elements.append({
                "section": section,
                "type": visual_suggestion.type,
                "data_points": visual_suggestion.data_points
            })
        except Exception as e:
            logger.error(f"Failed to parse LLM response as JSON in visual_creator for section '{section}': {getattr(visual_suggestion, 'content', str(visual_suggestion))}")
            visual_elements.append({
                "section": section,
                "type": "unknown",
                "data_points": []
            })
    return {
        "visual_elements": visual_elements
    }


def template_selector(state: ContentMasterState):
    # Choose appropriate template based on content type
    structured_llm = llm.with_structured_output(TemplateConfig, method="json_mode")
    try:
        template_selection = structured_llm.invoke(f"""
        Select the most appropriate template for:
        Content Type: {state.content_type}
        Main Topic: {state.research_query}
        
        Return your answer as a JSON object with fields 'layout', 'colors', 'typography', and 'visual_hierarchy', e.g.:
        {{
            "layout": {{"type": "grid", "columns": 2}},
            "colors": {{"primary": "#4682B4", "background": "#FFFFFF"}},
            "typography": {{"font-family": "Lato", "font-size": 16}},
            "visual_hierarchy": {{"heading": {{"font-size": 24}}}}
        }}
        """)
        return {
            "template": template_selection.dict()
        }
    except Exception as e:
        logger.error(f"Failed to parse LLM response as JSON in template_selector: {getattr(template_selection, 'content', str(template_selection))}")
        return {
            "template": {}
        }


def content_assembler(state: ContentMasterState):
    # Combine all elements into final output
    final_output = llm.invoke(f"""
    Assemble the final {state.content_type} with:
    - Generated Content
    - Visual Elements
    - Selected Template
    
    Include:
    - Proper formatting
    - Source citations
    - Coherent flow
    """)
    
    return {
        "final_output": final_output.content
    }


######################## Graph Workflow ########################

def build_content_master_graph():
    workflow = StateGraph(ContentMasterState)
    
    # Add nodes
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("source_verifier", source_verifier)
    workflow.add_node("content_planner", content_planner)
    workflow.add_node("content_generator", content_generator)
    workflow.add_node("visual_creator", visual_creator)
    workflow.add_node("template_selector", template_selector)
    workflow.add_node("content_assembler", content_assembler)
    
    # Define edges
    workflow.add_edge(START, "query_analyzer")
    workflow.add_edge("query_analyzer", "research_agent")
    workflow.add_edge("research_agent", "source_verifier")
    workflow.add_edge("source_verifier", "content_planner")
    workflow.add_edge("content_planner", "content_generator")
    workflow.add_edge("content_generator", "visual_creator")
    workflow.add_edge("visual_creator", "template_selector")
    workflow.add_edge("template_selector", "content_assembler")
    workflow.add_edge("content_assembler", END)


    # Conditional Edges for Quality Control
    def research_quality_check(state: ContentMasterState):
        # Determine if research results are sufficient
        if len(state.search_results) < 3:
            return "retry_research"
        return "proceed_to_verification"


    def source_verification_check(state: ContentMasterState):
        # Check source credibility
        if len(state.verified_sources) < 2:
            return "retry_research"
        return "proceed_to_planning"


    def visual_requirement_check(state: ContentMasterState):
        # Determine if visuals are needed based on content type
        if state.content_type in ["presentation", "webpage"]:
            return "create_visuals"
        return "skip_visuals"


    # Add Conditional Edges
    workflow.add_conditional_edges(
        "research_agent", 
        research_quality_check,
        {
            "retry_research": "query_analyzer",
            "proceed_to_verification": "source_verifier"
        }
    )

    workflow.add_conditional_edges(
        "source_verifier",
        source_verification_check,
        {
            "retry_research": "query_analyzer",
            "proceed_to_planning": "content_planner"
        }
    )

    workflow.add_conditional_edges(
        "content_generator",
        visual_requirement_check,
        {
            "create_visuals": "visual_creator",
            "skip_visuals": "template_selector"
        }
    )

    # Note: You will need nodes "retry_research", "create_visuals", "skip_visuals" defined appropriately
    # or you can adjust this logic to your graph setup.
    # For now, this example uses the primary linear edges defined above.

    # Compile the graph
    return workflow.compile()


# Create the graph
content_master_graph = build_content_master_graph()


######################## Execution Function ########################

def run_content_master(user_request: str):
    # Stream the graph execution first for intermediate outputs
    print("\n--- Intermediate Steps (from stream) ---")
    temp_initial_state_for_stream = ContentMasterState(user_request=user_request)
    try:
        for output in content_master_graph.stream(temp_initial_state_for_stream, stream_mode="updates"):
            for key, value in output.items():
                print(f"Node '{key}' updates:")
                print(value)
            print("\n---\n")
    except Exception as e:
        print(f"Error during streaming execution: {e}")
        return None

    # Then, get the final result by invoking cleanly with fresh state
    print("\n--- Final Result (from invoke) ---")
    try:
        final_result = content_master_graph.invoke(ContentMasterState(user_request=user_request))
        # Fix: handle both dict and object returns
        if isinstance(final_result, dict):
            return final_result.get("final_output")
        else:
            return getattr(final_result, "final_output", None)
    except Exception as e:
        print(f"Error in content generation: {e}")
        return None


######################## Main execution ########################

def main():
    # Different content type requests
    requests = [
        "Create a presentation on renewable energy trends",
        "Write a detailed document about AI ethics",
        "Generate a webpage about space exploration"
    ]

    for request in requests:
        print(f"\n--- Processing: {request} ---")
        try:
            output = run_content_master(request)

            if output:
                print("Generated Content:")
                print(output)

                # Optional: Save output to a file
                filename = f"output_{request.replace(' ', '_')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(output)
                print(f"Output saved to {filename}")
            else:
                print("Failed to generate content.")

        except Exception as e:
            print(f"Unexpected error processing request: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContentMaster: AI-Powered Content Generation")
    parser.add_argument(
        "--request", 
        type=str, 
        help="Specific content generation request"
    )

    args = parser.parse_args()

    if args.request:
        output = run_content_master(args.request)
        print(output)
    else:
        main()