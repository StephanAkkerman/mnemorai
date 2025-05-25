"use client"

import React, { useState } from "react"

const FAQSection = () => {
  // State to track which FAQ items are expanded
  const [expandedItems, setExpandedItems] = useState<{ [key: number]: boolean }>({});

  // FAQ data - doubled to 10 items
  const faqItems = [
    {
      question: "How effective are AI-generated mnemonics for language learning?",
      answer: "Mnemonic techniques significantly boost memory retention. Our AI tailors these techniques to you, creating stronger, personalized connections by linking new vocabulary to sounds, images, and concepts you already understand, making recall more intuitive than rote memorization."
    },
    {
      question: "How does the AI create personalized mnemonics?",
      answer: "mnemorai's AI analyzes the word's pronunciation, meaning, and potential cultural context. It considers your target language and finds phonetic links or conceptual associations to generate memorable images and phrases tailored to aid your recall."
    },
    {
      question: "Do I need Anki or AnkiConnect to use mnemorai?",
      answer: "While you can generate cards directly on our website, saving them to your personal collection requires the Anki desktop application and the AnkiConnect add-on to be installed and running on your computer. This allows seamless synchronization."
    },
    {
      question: "Can I edit the flashcards created by the AI?",
      answer: "Yes! After a card is generated, you can review and edit the word, IPA transcription, and mnemonic phrase directly within the mnemorai interface before saving it to Anki, or modify it later in your library."
    },
    {
      question: "What languages does mnemorai support?",
      answer: "mnemorai supports a growing list of languages for card generation. The available languages are dynamically fetched from our backend service when you use the card generator tool."
    },
    {
      question: "Can I use mnemorai on my phone?",
      answer: "mnemorai is designed to be responsive and accessible through the web browser on your desktop, tablet, or mobile device, allowing you to generate and review cards on the go."
    },
    {
      question: "How quickly can I expect to see results?",
      answer: "Many users report feeling more confident with new vocabulary within the first couple of weeks. Consistent use helps build strong memory associations, leading to significantly better long-term retention compared to traditional flashcard methods."
    },
    {
      question: "Can I choose different AI models for generation?",
      answer: "Yes, the 'Advanced Options' in the Card Generator allow you to select from available Large Language Models (LLMs) and Image Generation models to customize the card creation process."
    },
    {
      question: "Is there a free trial or free usage?",
      // Answer based on your intended model - assuming some free tier/trial
      answer: "We offer a way to try out the card generation features. Saving cards to Anki and accessing the full library may require a subscription or be part of a trial period. Please check our pricing page for details."
    },
    {
      question: "How do I cancel my subscription?",
      // Assuming a subscription model
      answer: "If you have a subscription, you can typically manage or cancel it anytime from your account settings page. Cancellation details depend on the specific plan and timing."
    }
  ];

  // Toggle expanded state for a specific FAQ item
  const toggleExpanded = (index: number) => {
    setExpandedItems(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return (
    <div className="container mx-auto px-6 ">
      <h2 className="text-3xl font-bold text-center text-gray-800 dark:text-white mb-8">
        Frequently Asked Questions
      </h2>

      {/* Scrollable FAQ container with custom scrollbar */}
      <div className="max-w-3xl mx-auto max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
        <div className="border-t border-gray-200">
          {faqItems.map((item, index) => (
            <div
              key={index}
              className="border-b border-gray-200"
            >
              <button
                onClick={() => toggleExpanded(index)}
                className="w-full flex justify-between items-center py-5 px-2 text-left focus:outline-none focus:ring-0"
              >
                <h3
                  className={`text-lg font-medium transition-colors duration-300 ${expandedItems[index]
                    ? 'bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent'
                    : 'text-gray-800 dark:text-white'
                    }`}
                >
                  {item.question}
                </h3>
                <svg
                  className={`w-5 h-5 transition-transform duration-300 ${expandedItems[index]
                    ? 'rotate-180 text-teal-400'
                    : 'text-gray-500 dark:text-white'
                    }`}
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>

              {/* Answer section with slide animation */}
              <div
                className={`overflow-hidden transition-all duration-300 ease-in-out ${expandedItems[index]
                  ? 'max-h-40 opacity-100 pb-5 px-2'
                  : 'max-h-0 opacity-0'
                  }`}
              >
                <p className="text-gray-600 dark:text-white">
                  {item.answer}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Custom scrollbar styling */}
      <style jsx global>{`
                /* Customize scrollbar for webkit browsers */
                .custom-scrollbar::-webkit-scrollbar {
                    width: 8px;
                }
                
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: transparent;
                }
                
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background-color: #9CA3AF; /* Gray-500 */
                    border-radius: 20px;
                }
                
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: linear-gradient(to bottom, #3B82F6, #2DD4BF); /* blue-500 to teal-400 */
                }
                
                /* For Firefox */
                .custom-scrollbar {
                    scrollbar-width: thin;
                    scrollbar-color: #9CA3AF transparent;
                }
                
                /* Make sure expanded answers can be taller for longer content */
                .max-h-40 {
                    max-height: 160px; /* Increased from default */
                }
            `}</style>
    </div>
  )
}

export default FAQSection
