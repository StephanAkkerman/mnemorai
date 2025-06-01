"use client";

import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { ANKI_CONFIG } from "@/config/constants";
import { useToast } from "@/contexts/ToastContext";

interface APIStatus {
  available: boolean | null;
  name: string;
  description: string;
}

export default function StatusChecker() {
  const pathname = usePathname();
  // Moved toast context hook up as it's needed even when on homepage to hide toasts
  const { showToast, hideAllToasts } = useToast();

  const statusesRef = useRef<APIStatus[]>([
    { name: "AnkiConnect", description: "Anki synchronization", available: null },
    { name: "Card Generator", description: "Card creation API", available: null },
  ]);

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isMounted = useRef(true); // To track if component is mounted for async operations

  // Define the API status check function (no changes here)
  const checkAPIStatus = async (name: string): Promise<boolean> => {
    try {
      if (name === "AnkiConnect") {
        const response = await fetch(ANKI_CONFIG.API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "version", version: 6 }),
        });
        if (response.ok) {
          const data = await response.json();
          return !!data.result;
        }
      }
      if (name === "Card Generator") {
        const response = await fetch("http://localhost:8000/create_card/supported_languages");
        if (response.ok) {
          return true;
        }
      }
      return false;
    } catch (error) {
      console.error(`Error checking ${name}:`, error);
      return false;
    }
  };

  // Effect for mount/unmount status
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      // General cleanup for interval if component fully unmounts
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, []);


  const POLLING_INTERVAL = 5000;
  const SUCCESS_DISPLAY_TIME = 3000;

  // Main effect for polling and toast logic, now dependent on pathname
  useEffect(() => {

    if (pathname === "/") {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      hideAllToasts(); // Crucial: hide toasts when navigating to or on the homepage
      return;
    }

    // --- Logic for non-homepage paths ---

    const checkAllStatuses = async () => {
      if (!isMounted.current) return;

      const currentStatuses = [...statusesRef.current];
      const updatedStatuses = [...currentStatuses];
      const recoveredServices: string[] = [];
      let hasServiceDown = false;
      let statusChanged = false;

      for (let i = 0; i < updatedStatuses.length; i++) {
        const status = updatedStatuses[i];
        const isAvailable = await checkAPIStatus(status.name);

        if (status.available === false && isAvailable) {
          updatedStatuses[i] = { ...status, available: true };
          recoveredServices.push(status.name);
          statusChanged = true;
        } else if (status.available === null && isAvailable) {
          updatedStatuses[i] = { ...status, available: true };
        } else if (!isAvailable) {
          if (status.available !== false) {
            statusChanged = true;
          }
          updatedStatuses[i] = { ...status, available: false };
          hasServiceDown = true;
        }
      }

      statusesRef.current = updatedStatuses;

      if (statusChanged) {
        hideAllToasts();

        if (recoveredServices.length > 0) {
          showToast({
            type: 'success',
            title: `Service${recoveredServices.length > 1 ? 's' : ''} Recovered`,
            message: `${recoveredServices.join(", ")} ${recoveredServices.length > 1 ? 'are' : 'is'} now available.`,
            duration: SUCCESS_DISPLAY_TIME
          });
        }

        if (hasServiceDown) {
          setTimeout(() => {
            if (!isMounted.current) return;
            const unavailableServices = statusesRef.current
              .filter(s => s.available === false)
              .map(s => s.name)
              .join(", ");
            if (unavailableServices) { // Only show if there are actually unavailable services
              showToast({
                type: 'error',
                title: 'Service Interruption',
                message: `${unavailableServices} ${statusesRef.current.filter(s => s.available === false).length > 1 ? 'are' : 'is'} unavailable. Please check your connections.`,
                duration: 0
              });
            }
          }, recoveredServices.length > 0 ? 300 : 0);
        } else if (recoveredServices.length > 0) {
          setTimeout(() => {
            if (!isMounted.current) return;
            showToast({
              type: 'info',
              title: 'All Systems Operational',
              message: 'All services are now running properly.',
              duration: SUCCESS_DISPLAY_TIME
            });
          }, 300);
        }
      }
    };

    // Run initial check only if not on homepage (already handled by the if (pathname === "/") check)
    checkAllStatuses();

    // Set up polling interval only if not already set and not on homepage
    if (!pollingIntervalRef.current) {
      pollingIntervalRef.current = setInterval(checkAllStatuses, POLLING_INTERVAL);
    }

    // Cleanup for this effect:

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }

    };
  }, [pathname, showToast, hideAllToasts]);

  if (pathname === "/") {
    return null;
  }

  return null;
}