import { streamGemini } from './gemini-api.js';

let form = document.querySelector('form');
let promptInput = document.querySelector('input[name="prompt"]');
let output = document.querySelector('.output');
let feedbackDiv = document.querySelector('.response-feedback');
let feedbackTextDiv = document.querySelector('.feedback-text');
let thumbsUpBtn = document.querySelector('.thumbs-up');
let thumbsDownBtn = document.querySelector('.thumbs-down');
let submitFeedbackBtn = document.getElementById('submit-feedback');
let thankYouMessageDiv = document.querySelector('.thank-you-message');

feedbackDiv.style.display = "none"; // Hide feedback initially

// Form submission to generate output
form.onsubmit = async (ev) => {
  ev.preventDefault();
  output.textContent = 'Generating...';

  try {
    let contents = [
      {
        role: 'user',
        parts: [
          { text: promptInput.value }
        ]
      }
    ];

    // Call the multimodal model, and get a stream of results
    let stream = streamGemini({
      model: 'gemini-1.5-flash', // or gemini-1.5-pro
      contents,
    });

    // Read from the stream and interpret the output as markdown
    let buffer = [];
    let md = new markdownit();
    for await (let chunk of stream) {
      buffer.push(chunk);
      output.innerHTML = md.render(buffer.join(''));
    }

    // Show feedback buttons after response is generated
    feedbackDiv.style.display = "block";
  } catch (e) {
    output.innerHTML += '<hr>' + e;
  }
};


thumbsUpBtn.addEventListener('click', () => {
  let feedbackData = {
      prompt: promptInput.value,
      response: document.querySelector('.output').innerHTML,
      feedback_type: "thumbs-up",
      additional_feedback: "N/A", // No additional feedback for thumbs-up
  };
  submitFeedback(feedbackData);

  feedbackDiv.style.display = "none"; // Hide thumbs buttons
  thankYouMessageDiv.style.display = "block"; // Clear and show the thank-you message
});

thumbsDownBtn.addEventListener('click', () => {
  feedbackDiv.style.display = "none"; // Hide thumbs buttons
  feedbackTextDiv.style.display = "block"; // Show feedback textarea
});

submitFeedbackBtn.addEventListener('click', () => {

  let feedbackData = {
      prompt: promptInput.value,  // User input prompt
      response: document.querySelector('.output').innerHTML,  // Model's response from the output element
      feedback_type: "thumbs-down",  // Feedback type: thumbs-up or thumbs-down
      additional_feedback: document.getElementById('feedback-textarea').value, // Get feedback text
  };

  submitFeedback(feedbackData);

  // Clear and show the thank-you message
  document.getElementById('feedback-textarea').value = '';
  feedbackTextDiv.style.display = "none";
  thankYouMessageDiv.style.display = "block";
});

async function submitFeedback(data) {
  try {
      let response = await fetch('/api/submit-feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
      });

      if (!response.ok) {
          throw new Error(await response.text());
      }

      console.log('Feedback submitted successfully!');
  } catch (error) {
      console.error('Error submitting feedback:', error);
  }
}
