"use client"
import WhatSection from "./landing-page/2-WhatSection"
import WhySection from "./landing-page/3-WhySection";
import HowSection from "./landing-page/4-HowSection";
import HeroSection from "./landing-page/1-HeroSection";
import AIPage from "./landing-page/5-AISection";
import FAQSection from "./landing-page/6-FAQSection";
import CTASection from "./landing-page/7-CTASection";
import React from "react";

const LandingPage = () => {
  return (
    <>
      <div className="min-h-screen flex flex-col w-full">
        <section id="home">
          <HeroSection />
        </section>

        <section id="what" className="duration-300 transition-all -translate-y-[10%] md:-translate-y-[15%]">
          <WhatSection />
        </section>

        {/* Full width section - breaks out of container */}
        <section id="why" className="duration-300 transition-all -translate-y-[5%] w-screen relative left-1/2 right-1/2 -mx-[50vw]  ">
          <WhySection />
        </section>

        {/* 2. HOW IT WORKS SECTION */}
        <section id="how">
          <HowSection />
        </section>

        {/* 3. FEATURES & BENEFITS SECTION - Full width */}
        <section id="features" className="py-20 w-screen relative left-1/2 right-1/2 -mx-[50vw]">
          <AIPage />
        </section>

        {/* 5. FAQ SECTION */}
        <section id="faq">
          <FAQSection />
        </section>

        {/* 6. CTA */}
        <section id="cta" className="pt-10" >
          <CTASection />
        </section>
      </div>
    </>
  );
};

export default LandingPage;
