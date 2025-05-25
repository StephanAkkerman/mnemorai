"use client"

import React from "react";



const Footer = () => {



    return (
        <>

            <footer className="bg-gray-800 text-white py-12">
                <div className="container mx-auto px-6">
                    <div className="flex flex-col md:flex-row justify-between mb-8">
                        <div className="mb-6 md:mb-0">
                            <div className="flex items-center mb-4">
                                <span className="text-xl font-bold">mnemorai</span>
                            </div>
                            <p className="text-gray-400 max-w-xs">
                                Learn languages faster with AI-powered mnemonic flashcards.
                            </p>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-3 gap-8">
                            <div>
                                <h3 className="text-lg font-semibold mb-4">Product</h3>
                                <ul className="space-y-2">
                                    <li><a href="#" className="text-gray-400 hover:text-white">Features</a></li>
                                    <li><a href="#" className="text-gray-400 hover:text-white">Pricing</a></li>
                                    <li><a href="#" className="text-gray-400 hover:text-white">Languages</a></li>
                                </ul>
                            </div>

                            <div>
                                <h3 className="text-lg font-semibold mb-4">Company</h3>
                                <ul className="space-y-2">
                                    <li><a href="#" className="text-gray-400 hover:text-white">About</a></li>
                                    <li><a href="#" className="text-gray-400 hover:text-white">Blog</a></li>
                                    <li><a href="#" className="text-gray-400 hover:text-white">Contact</a></li>
                                </ul>
                            </div>

                            <div>
                                <h3 className="text-lg font-semibold mb-4">Legal</h3>
                                <ul className="space-y-2">
                                    <li><a href="#" className="text-gray-400 hover:text-white">Terms</a></li>
                                    <li><a href="#" className="text-gray-400 hover:text-white">Privacy</a></li>
                                    <li><a href="#" className="text-gray-400 hover:text-white">Cookies</a></li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    <div className="border-t border-gray-700 pt-8">
                        <div className="flex flex-col md:flex-row justify-between items-center">
                            <p className="text-gray-400 text-sm mb-4 md:mb-0">
                                © 2025 mnemorai. All rights reserved.
                            </p>
                            <div className="flex space-x-4">

                            </div>
                        </div>
                    </div>
                </div>
            </footer>
        </>
    )
}

export default Footer;