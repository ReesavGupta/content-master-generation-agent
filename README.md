# ContentMaster

ContentMaster is a LangGraph-powered AI agent that combines Gamma's presentation creation and Perplexity's search capabilities to generate research-backed, professional content in multiple formats.

---

## 🚀 Features

- **Smart Research**: Web search with source verification for credible, up-to-date information.
- **Content Generation**: AI-powered creation of slides, documents, and web pages.
- **Visual Creation**: Automatic generation of charts, diagrams, and infographics.
- **Template Intelligence**: Dynamic layouts and templates based on content type.
- **Citation Management**: All facts and data are cited with verified sources.

---

## 🧩 LangGraph Workflow

### **State**

The agent maintains a state object (`ContentMasterState`) that tracks:
- User request
- Research query
- Search results
- Verified sources
- Content outline
- Generated content
- Visual elements
- Final output
- Content type

### **Nodes**

1. **Query Analyzer**: Parses the user request and determines the content type and research query.
2. **Research Agent**: Performs a web search and gathers information.
3. **Source Verifier**: Validates and ranks search results for credibility.
4. **Content Planner**: Creates a structured outline with sections and key points.
5. **Content Generator**: Generates text, headlines, and copy for each section.
6. **Visual Creator**: Suggests and generates charts, diagrams, and layouts.
7. **Template Selector**: Chooses the most appropriate design template.
8. **Content Assembler**: Combines all elements into the final output.

### **Edges**

```
START → Query Analyzer → Research Agent → Source Verifier → Content Planner → Content Generator → Visual Creator → Template Selector → Content Assembler → END
```

#### **Conditional Edges for Quality Control**
- After Research Agent: If insufficient results, retry with a refined query; else, proceed to Source Verifier.
- After Source Verifier: If low-quality sources, retry; else, proceed to Content Planner.
- After Content Generator: If visuals are needed, go to Visual Creator; else, skip to Template Selector.

---

## 📝 Example Usage

**Input:**  
`"Create a presentation on renewable energy trends"`

**Process:**  
Research → Verify sources → Plan slides → Generate content → Create visuals → Apply template

**Output:**  
A professional slide deck (or doc/webpage) with citations and visuals, saved as a text file.

---

## 🛠️ How to Run

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Set up your environment:**
   - Add your API keys (for Groq, Tavily, etc.) to a `.env` file.

3. **Run the script:**
   - For default demo:
     ```
     python main.py
     ```
   - For a custom request:
     ```
     python main.py --request "Write a detailed document about AI ethics"
     ```

4. **Output:**
   - Generated content will be printed and saved to a file named after your request (e.g., `output_Create_a_presentation_on_renewable_energy_trends.txt`).

---

## 🧪 Testing

The `main()` function includes tests for:
- Presentation generation
- Document writing
- Webpage creation

You can add more test cases by editing the `requests` list in `main.py`.

---

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

---

## 📄 License

MIT License

---

## 🤝 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Gamma](https://gamma.app/)
- [Perplexity](https://www.perplexity.ai/)
- [Tavily](https://www.tavily.com/)

---

## 💡 Notes

- The agent is designed for extensibility. You can add more nodes, templates, or content types as needed.
- All outputs include proper citations and, where appropriate, visual elements.

---

**ContentMaster**: Your AI-powered, research-backed content generator! 