
var selected1 = [];
var selected2 = [];
var resp;
var brush = null;
var globalData = null;
var graph1 = null;
var graph2 = null;
var highlightedGraphs = [];


function main() {
    getGlobalData(0);

    const btnLoad = document.getElementById('btn-load');
    btnLoad.addEventListener('click', function (e) {
        const inputGraph1 = document.getElementById('input-graph1');
        const inputGraph2 = document.getElementById('input-graph2');
        let graphName1 = inputGraph1.value;
        let graphName2 = inputGraph2.value;
        for (let i = 0; i < globalData.dataset.length; ++i) {
            if (globalData.dataset[i]['id'] == graphName1) {
                graph1 = i;
            }
            if (globalData.dataset[i]['id'] == graphName2) {
                graph2 = i;
            }
        }

        getData(graph1, graph2);
        drawGraphEmbeddings(globalData.embeddings.x, globalData.embeddings.y);
    });
}

function getGlobalData(value) {
    fetch('/global', { 
        method: 'POST', 
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({something: value})
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error('error occured');
            }

            return response.json();
        })
        .then((response) => {
            console.log(response);

            globalData = response;
        
            const datalistGraph = document.getElementById('datalist-graph');

            for (let i = 0; i < response.dataset.length; ++i) {
                let optionItem = document.createElement('option');
                optionItem.value = response.dataset[i].id;
                datalistGraph.appendChild(optionItem);
            }

            graph1 = 1; graph2 = 2;
            getData(graph1, graph2);
            const inputGraph1 = document.getElementById('input-graph1');
            const inputGraph2 = document.getElementById('input-graph2');
            inputGraph1.value = response.dataset[1].id;
            inputGraph2.value = response.dataset[2].id;

            drawGraphEmbeddings(response.embeddings.x, response.embeddings.y);
        });
}

function getData(graph1, graph2) {
    fetch('/something', { 
        method: 'POST', 
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({graph1, graph2})
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error('error occured');
            }

            return response.json();
        })
        .then((response) => {
            console.log(response);
            selected1 = response.data.labels_1.map((v) => true)
            selected2 = response.data.labels_2.map((v) => true)

            resp = response;
            
            drawNodeEmbeddings(response.x, response.y, response.data.labels_1);
            render();

            const predictedSpan = document.getElementById('span-predicted');
            predictedSpan.innerText = response.prediction.toString();
        });
}


function render() {
    drawGraph('#svg-graph1', resp.data.graph_1, resp.data.labels_1, resp.node_weight1, resp.edge_weight1, selected1);
    drawGraph('#svg-graph2', resp.data.graph_2, resp.data.labels_2, resp.node_weight2, resp.edge_weight2, selected2);

    const tableNode = document.getElementById('table-node');
    tableNode.innerHTML = "";
    let thItem = document.createElement('tr');
    let tdItem = document.createElement('td')
    tdItem.innerText = 'Id';
    thItem.appendChild(tdItem);
    tableNode.appendChild(thItem);
    tdItem = document.createElement('td')
    tdItem.innerText = 'Label';
    thItem.appendChild(tdItem);
    tableNode.appendChild(thItem);
    for (let i = 0; i < selected1.length; ++i) {
        if (selected1[i]) {
            let trItem = document.createElement('tr');
            let tdItem = document.createElement('td')
            tdItem.className = 'text-red';
            tdItem.innerText = i.toString();
            trItem.appendChild(tdItem);
            trItem.appendChild(tdItem);
            tableNode.appendChild(trItem);
            tdItem = document.createElement('td')
            tdItem.className = 'text-red';
            tdItem.innerText = resp.data.labels_1[i];
            trItem.appendChild(tdItem);
            tableNode.appendChild(trItem);
        }
    }
    for (let i = 0; i < selected2.length; ++i) {
        if (selected2[i]) {
            let trItem = document.createElement('tr');
            let tdItem = document.createElement('td')
            tdItem.className = 'text-blue';
            tdItem.innerText = i.toString();
            trItem.appendChild(tdItem);
            trItem.appendChild(tdItem);
            tableNode.appendChild(trItem);
            tdItem = document.createElement('td')
            tdItem.className = 'text-blue';
            tdItem.innerText = resp.data.labels_2[i];
            trItem.appendChild(tdItem);
            tableNode.appendChild(trItem);
        }
    }
   

}

function updateGraphSelection(e, x, y, xScale, yScale) {
    const ulGraph = document.getElementById('ul-graph');
    ulGraph.innerHTML = '';

    if (e.selection == null ) {
        return ;
    }
    e = e.selection;

    let x0 = e[0][0];
    let y0 = e[0][1];
    let x1 = e[1][0];
    let y1 = e[1][1];
    
    highlightedGraphs = globalData.dataset.filter((v, i) => {
        return x0 <= xScale(x[i]) && xScale(x[i]) <= x1 && y0 <= yScale(y[i]) && yScale(y[i]) <= y1
    })

    for (let i = 0; i < highlightedGraphs.length; ++i) {
        let liItem = document.createElement('li');
        liItem.innerText = highlightedGraphs[i].id;
        ulGraph.appendChild(liItem);
    }
}

function updateSelection(e, x, y, xScale, yScale) {
    if (e.selection == null ) {
        e = [[0, 0], [300, 300]];
    }
    else {
        e = e.selection;
    }
    let x0 = e[0][0];
    let y0 = e[0][1];
    let x1 = e[1][0];
    let y1 = e[1][1];

    selected1 = selected1.map((v, i) => {
        return x0 <= xScale(x[i]) && xScale(x[i]) <= x1 && y0 <= yScale(y[i]) && yScale(y[i]) <= y1
    })

    selected2 = selected2.map((v, i) => {
        return x0 <= xScale(x[i + selected1.length]) && xScale(x[i + selected1.length]) <= x1 && y0 <= yScale(y[i + selected1.length]) && yScale(y[i + selected1.length]) <= y1
    })

    render();
}

function drawNodeEmbeddings(x, y, labels_1) {
    let width = 300;
    let height = 300;

    let divNode = d3.select('#div-node');
    divNode.selectAll('*').remove();


    let svg = divNode.append('svg')
        .attr('id', 'svg-node')
        .attr('width', width)
        .attr('height', height);

    let label = divNode.append('label')
        .attr('for', 'svg-node')
        .attr('class', 'svg-label')
        .text('Node Embedding');
        

    svg.append('rect')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('fill', 'rgb(230, 230, 230)');

    var xScale = d3.scaleLinear()
        .domain([d3.min(x), d3.max(x)])
        .range([20, 280]);

    var yScale = d3.scaleLinear()
        .domain([d3.min(y), d3.max(y)])
        .range([20, 280]);

        svg.call(d3.brush()
            .extent([[0, 0], [width, height]])
            .on('end', (e) => updateSelection(e, x, y, xScale, yScale)));


    console.log(x)

    let gdots = svg.selectAll('g.dot')
        .data(x)
        .enter().append('g');


    gdots.append("circle")
        .attr("cx", function (d, i) { return xScale(x[i]); })
        .attr("cy", function (d, i) { return yScale(y[i]); })
        .attr("r", 6)
        .style("fill", (d, i) => i < labels_1.length ? 'red' : 'blue')

}

function drawGraphEmbeddings(x, y) {
    let width = 300;
    let height = 300;

    let svg = d3.select('#svg-graph')
        .attr('width', width)
        .attr('height', height);
        
        if (!brush) {
        svg.selectAll('*').remove();
        
        let mainG = svg.append('g');
    
        svg.call(d3.brush()
            .extent([[0, 0], [width, height]])
            .on('end', (e) => updateGraphSelection(e, x, y, xScale, yScale)));
        brush = true;
    }
    let mainG = svg.select('g');

    var xScale = d3.scaleLinear()
            .domain([d3.min(x), d3.max(x)])
            .range([20, 280]);

    var yScale = d3.scaleLinear()
        .domain([d3.min(y), d3.max(y)])
        .range([20, 280]);

    let gdots = mainG.selectAll('dot')
        .data(x)
        .enter()
        .append("circle")
        .attr("cx", function (d, i) { return xScale(x[i]); })
        .attr("cy", function (d, i) { return yScale(y[i]); })
        .attr("r", 6)
        .style("fill", (d, i) => i == graph1 ? 'red' : ( i == graph2 ? 'blue' : 'gray'))


}

function drawGraph(elementId, edges, nodes, nodeWeights, edgeWeights, selected) {
    let width = 480;
    let height = 480;

    let nodes1 = []
    for (let i = 0; i < nodes.length; ++i) {
        nodes1.push({id: i, label: nodes[i]});
    }

    let links = [];
    for (let i = 0; i < edges.length; ++i) {
        let edge = edges[i];
        links.push({target: edge[1], source: edge[0], strength: 0.3});
    }
    
    let svg = d3.select(elementId)
        .attr('width', width)
        .attr('height', height);

    svg.append('rect')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('fill', 'rgb(230, 230, 230)');

    let mainG = svg.append('g');

    const simulation = d3.forceSimulation()
        .force('charge', d3.forceManyBody().strength(-130)) 
        .force('center', d3.forceCenter(width / 2, height / 2));

    let thickness = d3.scaleLinear().domain(d3.extent(edgeWeights)).range([2, 10])
        
    let linkElements = mainG.append('g')
        .selectAll('line')
        .data(links)
        .enter().append('line')
        .attr('stroke-width', (d, i) => thickness(edgeWeights[i]))
        .attr('stroke', 'rgb(0, 120, 250)');
    
    let nodeElements = mainG.append('g')
        .selectAll('circle')
        .data(nodes1)
        .enter().append('circle')
            .attr('r', 15)
            .attr('fill', (d, i) => {
                if (selected[i]) {
                    return d3.interpolateReds(nodeWeights[i]);
                }
                else {
                    return 'lightgreen';
                }
            })
            .attr('stroke', 'black');

    let textElements = mainG.append('g')
        .selectAll('text')
        .data(nodes1)
        .enter().append('text')
            .text(node => node.label)
            .attr('font-size', 20)
            .attr('dx', 0)
            .attr('dy', 0)
            .attr('class', 'text-something')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('stroke', 'grey')
            .attr('stroke-weight', '1')
        
    simulation.nodes(nodes1).on('tick', () => {
        nodeElements
            .attr('cx', node => node.x)
            .attr('cy', node => node.y)
        textElements
            .attr('x', node => node.x)
            .attr('y', node => node.y)
        linkElements
            .attr('x1', link => link.source.x)
            .attr('y1', link => link.source.y)
            .attr('x2', link => link.target.x)
            .attr('y2', link => link.target.y)
    })

    simulation.force('link', d3.forceLink()
        .id(link => link.id)
        .strength(link => link.strength));

    simulation.force('link').links(links);

    // function handleZoom(e) {
    //     d3.select('svg g')
    //         .attr('transform', e.transform);
    // }

    let zoom = d3.zoom()
        .on('zoom', function (e) {
            mainG.attr('transform', e.transform)
        });

    svg.call(zoom);
}


window.onload = main;