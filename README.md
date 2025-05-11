# Void

Void is an advanced AI personal assistant with memory capabilities, procrastination monitoring, and various tool integrations to enhance productivity and personal organization.

## Features

- **Memory System**: Long-term memory with episodic, semantic, and procedural components
- **Procrastination Monitoring**: Analyzes screen activity and helps keep you on task
- **Voice Interface**: Natural conversation through speech recognition and synthesis
- **Tool Integrations**:
  - Spotify: Control music playback
  - Gmail: Read and manage emails
  - Notion: Task management and note-taking
  - Google Calendar: Schedule management
  - Twitter: Social media engagement
  - Web Search: Information retrieval

## Requirements

- Python 3.9+
- API keys for various services:
  - Google Gemini API
  - Groq API (for some components)
  - Various tool-specific API keys (Spotify, Notion, etc.)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Void.git
   cd Void
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables by creating a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   NOTION_API_KEY=your_notion_api_key
   # Add other necessary API keys
   ```

5. Set up Google API credentials:
   - Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the Gmail API and Google Calendar API
   - Create OAuth 2.0 credentials and download the JSON file
   - Rename the downloaded file to `credentials2.json` and place it in the project root
   - A template file `credentials2.json.example` is provided for reference

## Usage

1. Start the interactive chat interface:
   ```
   python agent.py
   ```

2. For voice interaction:
   ```
   # Inside the agent.py interface, use the voice commands feature
   ```

3. For procrastination monitoring:
   ```
   python procrastination_monitor.py
   ```

## Architecture

- **agent.py**: Main assistant interface and agent orchestration
- **memoryprocessor.py**: Long-term memory system implementation
- **procrastination_monitor.py**: Screen activity analysis for task monitoring
- **screen.py**: Screen analysis tools
- **keyboard_tools.py**: Keyboard control utilities
- **Tool integrations**: Various modules for external service integrations
  - spotify.py, gmail_tools.py, notion_tools.py, calendar_tools.py, etc.

## Security Note

This project requires access to sensitive APIs and may monitor screen activity. All data is stored locally by default and not shared with external services beyond the necessary API calls.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See the [LICENSE](LICENSE) file for details.