function extractDOM() {
    function processNode(node) {
      const nodeData = {
        type: node.nodeName,
        children: [],
      };
  
      if (node.nodeType === Node.TEXT_NODE) {
        const trimmedText = node.textContent.trim();
        if (trimmedText) { // Avoid empty text nodes
          nodeData.text = trimmedText;
        }
      }
      if (node.nodeType === Node.ELEMENT_NODE) {
          if (node.hasAttributes()) {
              nodeData.attributes = {};
              for (let i = 0; i < node.attributes.length; i++) {
                  const attr = node.attributes[i];
                  nodeData.attributes[attr.name] = attr.value;
              }
          }
      }
  
      for (const child of node.childNodes) {
        const childData = processNode(child);
          if (childData) { // for discarding empty text node
              nodeData.children.push(childData);
          }
      }
      return (nodeData.text || nodeData.children.length > 0 || nodeData.attributes) ? nodeData : null;
    }
  
    return processNode(document.documentElement);
  }
  
  // Initial extraction
  chrome.runtime.sendMessage({ type: "DOM_UPDATE", dom: extractDOM() });
  
  // Observe changes
  const observer = new MutationObserver(() => {
    chrome.runtime.sendMessage({ type: "DOM_UPDATE", dom: extractDOM() });
  });
  
  observer.observe(document.documentElement, {
    subtree: true, // Watch all descendants
    childList: true, // Watch for additions/removals of children
    attributes: true, // Watch for attribute changes
    characterData: true, //Watch for text change
  });