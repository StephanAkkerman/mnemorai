"use client"
import React, { useRef, useEffect, useState } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import Button from "../ui/Button";

interface AIFeature {
  icon: string;
  title: string;
  description: string;
}

const AIPage: React.FC = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const [shouldShowContent, setShouldShowContent] = useState(false);

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start end", "end start"]
  });

  const [windowSize, setWindowSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const update = () => {
      if (typeof window !== "undefined") {
        setWindowSize({ width: window.innerWidth, height: window.innerHeight });
      }
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const isSmallScreen = windowSize && windowSize.width < 640;
  const isMediumScreen = windowSize && windowSize.width >= 640 && windowSize.width < 1024;

  const titleScale = useTransform(scrollYProgress, [0.2, 0.3], [1.2, 1]);
  const titleOpacity = useTransform(scrollYProgress, [0.2, 0.3], [0, 1]);
  const titleY = useTransform(
    scrollYProgress,
    [0.2, 0.3, 0.4, 0.5],
    isSmallScreen ? ["350%", "200%", "100%", "0%"] :
      isMediumScreen ? ["300%", "150%", "150%", "0%"] :
        ["300%", "150%", "150%", "0%"]
  );

  useEffect(() => {
    const updateVideoTime = (value: number) => {
      if (!videoRef.current) return;

      if (videoRef.current.duration && !isNaN(videoRef.current.duration)) {
        const newTime = value * videoRef.current.duration;
        if (Math.abs(videoRef.current.currentTime - newTime) > 0.5) {
          videoRef.current.currentTime = newTime;
        }
      }
      setShouldShowContent(value >= 0.45);
    };

    const unsubscribe = scrollYProgress.onChange(updateVideoTime);
    return () => unsubscribe();
  }, [scrollYProgress]);

  const overlayOpacity = useTransform(scrollYProgress, [0.3, 0.5, 0.7], [0.7, 0.8, 0.9]);

  // Animation Variants
  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } }
  };
  // Stagger container for the grid items
  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15
      }
    }
  };
  const fadeInLeft = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0, transition: { duration: 0.5, ease: "easeOut" } }
  };
  const fadeInRight = {
    hidden: { opacity: 0, x: 20 },
    visible: { opacity: 1, x: 0, transition: { duration: 0.5, ease: "easeOut" } }
  };

  const aiFeatures: AIFeature[] = [
    { icon: "🔍", title: "Smart Mnemonic Generation", description: "Our AI creates memory-boosting associations tailored to how your brain learns" },
    { icon: "🔊", title: "Phonetic Matching", description: "Discover similar-sounding words that bridge your native language with new vocabulary" },
    { icon: "🖼️", title: "Visual Learning Anchors", description: "Custom images that connect new words to concepts you already understand" },
    { icon: "🧠", title: "Personalized Learning Path", description: "AI adapts to your progress, focusing on what you need most" }
  ];

  return (
    <section
      ref={sectionRef}
      className="relative w-full h-[300vh]"
    >
      <div className="sticky top-0 left-0 w-full min-h-[130vh] sm:min-h-[120vh] h-screen flex items-center overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 w-full h-full">
          <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover" preload="auto" muted playsInline>
            <source src="/tech.mp4" type="video/mp4" />
            <img src="/tech.mp4" alt="AI technology" className="absolute inset-0 w-full h-full object-cover" />
          </video>
          <motion.div className="absolute inset-0 w-full h-full bg-black" style={{ opacity: overlayOpacity }} />
        </div>

        {/* Content Container */}
        <div className="relative z-10 w-full max-w-6xl mx-auto px-4 ">
          {/* Header */}
          <motion.div className="text-center" style={{ scale: titleScale, opacity: titleOpacity, y: titleY }}>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 text-white">
              Language Learning <span className="gradient-text">Powered by AI</span>
            </h2>
            <div className="h-1 w-20 bg-gradient-to-r from-blue-500 to-teal-400 mx-auto mt-2 mb-6 rounded-full"></div>
          </motion.div>

          {/* Main Content */}
          <motion.div
            className="w-full flex flex-col items-center flex-grow py-4 pt-0"
            initial="hidden"
            animate={shouldShowContent ? "visible" : "hidden"}
            variants={fadeInUp}
          >
            {/* How It Works */}
            {!isMediumScreen && !isSmallScreen &&
              <motion.div
                className="bg-gray-800/60 border border-gray-700 rounded-xl p-5 md:p-6 mb-6 md:mb-8 max-w-3xl mx-auto shadow-lg backdrop-blur-sm"
              >
                <h3 className="text-xl md:text-2xl font-semibold text-white mb-3">How Mnemorai Works</h3>
                <p className="text-gray-200 mb-3 text-base md:text-lg">
                  Our AI system builds powerful memory connections by finding words in your native language that sound similar to new vocabulary. These phonetic bridges, combined with vivid imagery, create lasting memory anchors that make recall effortless.
                </p>
                <p className="text-gray-200 text-base">
                  Unlike traditional flashcards that rely on repetition alone, Mnemorai creates meaningful associations that work with your brain&apos;s natural ability to remember stories and images.
                </p>
              </motion.div>}

            {/* AI Features Grid */}
            <motion.div
              className="w-full max-w-4xl"
              variants={staggerContainer}
            >
              <div
                className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5 md:gap-6 items-stretch justify-items-center"
              >
                {aiFeatures.map((feature, index) => (
                  <motion.div
                    key={index}
                    className="bg-gray-800/70 dark:bg-gray-900/80 backdrop-blur-sm border border-gray-700 dark:border-gray-600 rounded-xl p-4 sm:p-5 shadow-md hover:border-gray-500 transition-colors duration-300 flex flex-col h-full w-full max-w-md md:max-w-none"
                    variants={index % 2 === 0 ? fadeInLeft : fadeInRight}
                    custom={index}
                    whileHover={{ y: -4 }}
                  >
                    <div className="flex items-center mb-2 sm:mb-3">
                      <div className="text-xl sm:text-2xl mr-3 p-2 bg-gray-700/50 dark:bg-gray-800/60 rounded-lg">{feature.icon}</div>
                      <h3 className="text-lg sm:text-xl font-semibold text-white">{feature.title}</h3>
                    </div>
                    <p className="text-gray-300 dark:text-gray-400 text-sm sm:text-base flex-grow">
                      {feature.description}
                    </p>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* CTA Button */}
            <motion.div
              className="mt-4 md:mt-10 text-center"
            >
              <Button
                text="Try Mnemorai Now"
                variant="primary"
                link="https://demo.mnemorai.com/card-gen"
                className="px-6 py-2.5 sm:px-8 sm:py-3 text-base sm:text-lg"
              />
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default AIPage;
