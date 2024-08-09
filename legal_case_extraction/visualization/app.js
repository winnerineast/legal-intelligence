const app = Vue.createApp({
    data() {
        return {
            jsonData: null,
            tooltip: {
                visible: false,
                x: 0,
                y: 0,
                data: null
            },
            zoomScrollbarPosition: 0.387,
            svg: null, // Added to store reference to svg element
        };
    },
    mounted() {
        this.svg = d3.select("svg"); // Store reference to svg element
        // Initially render graph with sample data or wait for file input
    },
    methods: {
        handleFileChange(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = () => {
                try {
                    this.jsonData = JSON.parse(reader.result);
                    this.renderGraph();
                } catch (error) {
                    console.error('Error parsing JSON file:', error);
                }
            };
            reader.readAsText(file);
        },
        renderGraph() {
            const jsonData = this.jsonData;
            if (!jsonData) return;

            const width = 960;
            const height = 600;

            const canvas_x_limit = 2 * width;
            const canvas_y_limit = 2 * height;

            const svg = this.svg
                .attr("width", width)
                .attr("height", height);

            /* legend */
            // Legend Data
            const legendData = [
                { type: "Fact", shape: "triangle", color: "gray" },
                { type: "Law", shape: "rectangle", color: "gray" },
                { type: "Decision", shape: "circle", color: "gray" }
            ];

            // Create a legend
            const legend = svg.append("g")
                .attr("class", "legend")
                .attr("transform", "translate(20, 20)");  // Position the legend

            const legendItem = legend.selectAll(".legend-item")
                .data(legendData)
                .enter().append("g")
                .attr("class", "legend-item")
                .attr("transform", (d, i) => `translate(0, ${i * 30})`);  // Space out the legend items

            // Draw legend symbols
            legendItem.each(function(d) {
                const g = d3.select(this);
                if (d.shape === "rectangle") {
                    g.append("rect")
                        .attr("width", 20)
                        .attr("height", 20)
                        .attr("fill", d.color);
                } else if (d.shape === "triangle") {
                    g.append("path")
                        .attr("d", d3.symbol().type(d3.symbolTriangle).size(400))
                        .attr("fill", d.color)
                        .attr("transform", "translate(10, 15)");  // Adjust triangle position
                } else if (d.shape === "circle") {
                    g.append("circle")
                        .attr("r", 10)
                        .attr("cx", 10)
                        .attr("cy", 10)
                        .attr("fill", d.color);
                }
            });

            // Add labels to the legend symbols
            legendItem.append("text")
                .attr("x", 30)
                .attr("y", 15)
                .text(d => d.type)
                .attr("text-anchor", "start")
                .attr("alignment-baseline", "central");

        /* end of legend */

            const graphContainer = svg.append("g")
                .style("overflow", "scroll");

            const simulation = d3.forceSimulation()
                .force("link", d3.forceLink().id(d => d.id).strength(1))
                .force("charge", d3.forceManyBody().strength(-10))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collide", d3.forceCollide().radius(10))
                .force("friction", d3.forceManyBody().strength(5));
            
            const minZoom = 0.2;
            const maxZoom = 5;

            const zoom = d3.zoom()
                .scaleExtent([minZoom, maxZoom])
                .on("zoom", (event) =>{
                    const { x, y, k } = event.transform;
                    graphContainer.attr("transform", `translate(${x},${y}) scale(${k})`);
                    this.zoomScrollbarPosition = Math.log(k + 1) / Math.log(maxZoom + 1);
                    // console.log(this.zoomScrollbarPosition);
                });

            svg.call(zoom);

            const filter = svg.append("filter")
                .attr("id", "glow")
                .append("feGaussianBlur")
                .attr("stdDeviation", "2.5")
                .attr("result", "coloredBlur");

            graphContainer.append("rect")
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", canvas_x_limit)
                .attr("height", canvas_y_limit)
                .style("fill", "none")
                .style("stroke", "#7766cc")
                .style("stroke-width", 2)
                .style("filter", "url(#glow)");

            const nodes = [];
            const links = [];

            const item = jsonData;
            // console.log(item)

            Object.keys(item.facts).forEach(fact => {
                nodes.push({ id: fact, node_type: "fact", data: item.facts[fact] });
            });

            Object.keys(item.laws).forEach(law => {
                nodes.push({ id: law, node_type: "law", data: item.laws[law] });
            });

            Object.keys(item.decisions).forEach(decision => {
                nodes.push({ id: decision, node_type: "decision", data: item.decisions[decision] });
            });

            item.relations.forEach(relation => {
                // TODO: Handle "Not in the list" case
                if (relation.id1 === null || relation.id2 === null){
                    // Show a beautiful popup to warn the user 
                    alert("'Not in the list's are not allowed in the graph. Is the graph refined and complete?");
                    return;
                }

                links.push({
                    source: relation.id1,
                    target: relation.id2,
                    relationSummary: relation.relation_summary
                });
            });

            const link = graphContainer.selectAll(".link")
                .data(links)
                .enter().append("line")
                .attr("class", "link");

            function createCircleNode(d) {
                var element = document.createElementNS(d3.namespaces.svg, "circle");
                element.setAttribute("r", 8);
                return element;
            }
            
            function createRectNode(d) {
                var element = document.createElementNS(d3.namespaces.svg, "rect");
                element.setAttribute("width", 16);
                element.setAttribute("height", 16);
                // Center the rect element to the edges
                element.setAttribute("transform", "translate(-8, -8)");

                return element;
            }

            function createTriangleNode(d) {
                var groupElement = document.createElementNS(d3.namespaces.svg, "g");
                var polygonElement = document.createElementNS(d3.namespaces.svg, "polygon");
            
                const size = 20;  // Slightly larger for visual clarity
                const height = size * Math.sqrt(3) / 2;
                polygonElement.setAttribute("points", `0,${-height * 2 / 3} ${-size / 2},${height / 3} ${size / 2},${height / 3}`);
            
                // Append the <polygon> to the <g> element
                groupElement.appendChild(polygonElement);
            
                // Optionally set initial position using transform on the <g> element
                // groupElement.setAttribute("transform", "translate(0, 0)");
            
                return groupElement;
            }
            
            // Select and bind data to nodes
            const graphNodes = graphContainer.selectAll(".node")
                .data(nodes)
                .enter().append(d => {
                    if (d.node_type === "law") {
                        // For Laws
                        return createRectNode(d);
                    }
                    else if (d.node_type === "fact") {
                        // For Facts
                        return createTriangleNode(d);
                    }
                    else if (d.node_type === "decision") {
                        // For Decisions
                        return createCircleNode(d);
                    }
                })
                .attr("class", d => `node ${d.data.type}`)
                .on("mouseover", (event, d) => {
                    this.showTooltip(event.pageX, event.pageY, d);
                    d3.select(event.target).classed("hovered", true);
                })
                .on("mouseout", (event, d) => {
                    this.hideTooltip();
                    d3.select(event.target).classed("hovered", false);
                })
                .call(drag(simulation));
                

            graphNodes.append("title")
                .text(d => d.id);

            const label = graphContainer.selectAll(".label")
                .data(nodes)
                .enter().append("text")
                .attr("class", "label")
                .text(d => d.id)
                .attr("font-size", "10px")
                .attr("pointer-events", "none");

            simulation.nodes(nodes)
                .on("tick", ticked);

            simulation.force("link")
                .links(links);

            function ticked() {
                link.attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                // Update nodes
                graphNodes.each(function(d) {
                    if (d.node_type === "law") {
                        // For rect elements
                        d3.select(this)
                            .attr("x", d => d.x = Math.max(8, Math.min(canvas_x_limit - 8, d.x)))
                            .attr("y", d => d.y = Math.max(8, Math.min(canvas_y_limit - 8, d.y)));
                    } else if (d.node_type === "fact") {
                        // For <g> elements containing polygons
                        d3.select(this)
                            .attr("x", d => d.x = Math.max(8, Math.min(canvas_x_limit - 8, d.x)))
                            .attr("y", d => d.y = Math.max(8, Math.min(canvas_y_limit - 8, d.y)))
                            .attr("transform", d => {
                                let x = Math.max(8, Math.min(canvas_x_limit - 10, d.x));
                                let y = Math.max(8, Math.min(canvas_y_limit - 10, d.y));
                                return `translate(${x},${y})`;
                            });
                    } else if (d.node_type === "decision") {
                        // For circle elements
                        d3.select(this)
                            .attr("cx", d => d.x = Math.max(8, Math.min(canvas_x_limit - 8, d.x)))
                            .attr("cy", d => d.y = Math.max(8, Math.min(canvas_y_limit - 8, d.y)));
                    }
                });

                label.attr("x", d => d.x + 12)
                    .attr("y", d => d.y - 12);
            }

            function drag(simulation) {
                function dragstarted(event, d) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }

                function dragged(event, d) {
                    d.fx = event.x;
                    d.fy = event.y;
                }

                function dragended(event, d) {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }

                return d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended);
            }
        },
        showTooltip(x, y, data) {
            this.tooltip = { visible: true, x, y, data: data };
        },
        hideTooltip() {
            this.tooltip = { visible: false, x: 0, y: 0, data: null };
        }
    }
});

app.mount('#app');