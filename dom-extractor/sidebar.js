function displayDOM(domData, container) {
    container.innerHTML = ''; // Clear previous content

    function createNodeElement(nodeData) {
        const element = document.createElement('div');
        element.classList.add('node');

        const typeSpan = document.createElement('span');
        typeSpan.classList.add('node-type');
        typeSpan.textContent = nodeData.type;
        element.appendChild(typeSpan);

        if (nodeData.text) {
            const textSpan = document.createElement('span');
            textSpan.classList.add('node-text');
            textSpan.textContent = `"${nodeData.text}"`;
            element.appendChild(textSpan);
        }
        if (nodeData.attributes) {
            const attributesDiv = document.createElement('div');
            attributesDiv.classList.add('node-attributes');
            for (const key in nodeData.attributes) {
                const attrSpan = document.createElement('span');
                attrSpan.textContent = `${key}="${nodeData.attributes[key]}"`;
                attributesDiv.appendChild(attrSpan);
                attributesDiv.appendChild(document.createElement('br'));
            }
            element.appendChild(attributesDiv);
        }

        if (nodeData.children && nodeData.children.length > 0) {
            const childrenContainer = document.createElement('div');
            childrenContainer.classList.add('node-children');
            for (const child of nodeData.children) {
                childrenContainer.appendChild(createNodeElement(child));
            }
            element.appendChild(childrenContainer);
        }

        return element;
    }

    if (domData) {
      container.appendChild(createNodeElement(domData));
    }
}
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "UPDATE_SIDEBAR") {
    displayDOM(message.dom, document.getElementById('dom-tree'));
  }
});