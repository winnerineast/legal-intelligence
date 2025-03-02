chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "DOM_UPDATE") {
      // Send the DOM data to the sidebar
      chrome.runtime.sendMessage({
        type: "UPDATE_SIDEBAR",
        dom: message.dom
      });
    }
  });
  
  chrome.action.onClicked.addListener((tab) => {
      chrome.sidePanel.open({ tabId: tab.id });
  });