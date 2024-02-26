document.addEventListener('DOMContentLoaded', function() {
    const mapContainer = document.getElementById('game-map');

    // Function to refresh and draw the map
    function refreshMap() {
        fetch('http://localhost:8082/map/full/matrix')
            .then(response => response.json())
            .then(mapState => {
                mapContainer.innerHTML = ''; // Clear existing map
                mapState.forEach(row => {
                    row.forEach(tile => {
                        const tileElement = document.createElement('div');
                        tileElement.classList.add('tile');
                        let imageUrl;
                        switch(tile) {
                            // case 'A': imageUrl = 'path/to/apple-image.png'; break;
                            // case 'B': imageUrl = 'path/to/bones-image.png'; break;
                            // case 'J': imageUrl = 'path/to/jewel-image.png'; break;
                            // case 'G': imageUrl = 'path/to/grass-image.png'; break;
                            // case 'L': imageUrl = 'path/to/lump-image.png'; break;
                            // case 'R': imageUrl = 'path/to/rice-image.png'; break;
                            // case 'S': imageUrl = 'path/to/stone-image.png'; break;
                            // case 'W': imageUrl = 'path/to/wood-image.png'; break;
                            case '>': imageUrl = 'path/to/hill-image.png'; break;
                            case '<': imageUrl = 'path/to/revealed-hill-image.png'; break;
                            case '|': imageUrl = 'path/to/gate-image.png'; break;
                            case '$': imageUrl = 'path/to/mountain-image.png'; break;
                            case '_': imageUrl = 'path/to/meadow-image.png'; break;
                            case '.': imageUrl = 'path/to/revealed-meadow-image.png'; break;
                            case '-': imageUrl = 'path/to/water-image.png'; break;
                            case '+': imageUrl = 'path/to/forest-image.png'; break;
                            case ':': imageUrl = 'path/to/revealed-forest-image.png'; break;
                            case 'V': imageUrl = 'path/to/farmer-image.png'; break;
                            case 'B': imageUrl = 'path/to/bandit-image.png'; break;
                            case 'M': imageUrl = 'path/to/trader-image.png'; break;
                            case 'P': imageUrl = 'path/to/player-image.png'; break;
                            default: imageUrl = 'path/to/default-image.png';
                        }
                        tileElement.style.backgroundImage = `url('${imageUrl}')`;
                        mapContainer.appendChild(tileElement);
                    });
                });
            })
            .catch(error => console.error('Error:', error));
    }
        
    function refreshInventory() {
        // Fetch and display inventory items
        fetch('http://localhost:8082/player/inventory')
            .then(response => response.json())
            .then(inventory => {
                const inventoryItems = document.getElementById('inventory-items');
                if (Object.keys(inventory).length === 0) {
                    inventoryItems.textContent = 'Inventory is empty';
                } else {
                    inventoryItems.textContent = Object.entries(inventory)
                        .map(([item, quantity]) => `${item}: ${quantity}`)
                        .join(', ');
                }
            }).catch(error => console.error('Error:', error));;
    
        // Fetch and display total inventory value
        fetch('http://localhost:8082/player/inventory/value')
            .then(response => response.json())
            .then(value => {
                document.getElementById('inventory-value').textContent = value;
            }).catch(error => console.error('Error:', error));;
    
        // Fetch and display current gold
        fetch('http://localhost:8082/player/inventory/gold')
            .then(response => response.json())
            .then(gold => {
                document.getElementById('inventory-gold').textContent = gold;
            }).catch(error => console.error('Error:', error));
    }

    let over = false;
    function checkGameOver() {
        fetch('http://localhost:8082/map/isover')
        .then(response => response.json())
        .then(isOver => {
                if (isOver && !over) {
                    displayRestartPopup();
                    over = true;
                }
            }).catch(error => console.error('Error:', error));
    }
    
    function displayRestartPopup() {
        const popup = document.createElement('div');
        popup.className = 'popup-container';
        popup.innerHTML = `
            <div class="popup-header">Game Over</div>
            <div>
                <label for="mapNumber">Enter Map Number:</label>
                <input type="number" id="mapNumber" class="popup-input" value="1" min="0">
            </div>
            <button id="restartButton">Restart</button>
        `;
        document.body.appendChild(popup);
        
        document.getElementById('restartButton').addEventListener('click', () => {
            const mapNumber = document.getElementById('mapNumber').value;
            fetch(`http://localhost:8082/map/restart?map_number=${mapNumber}`, { method: 'PUT' })
            .then(() => {
                popup.remove();
                // reload();
                over = false;
                // You can add additional logic here to reset the game view
            }).catch(error => console.error('Error:', error));
        });
    }

    function reload()
    {
        // checkGameOver()
        refreshMap();
        refreshInventory();
    }

    reload();
    setInterval(reload, 75);     
    // setInterval(checkGameOver, 1000);
    
    // Keyboard event listener
    document.addEventListener('keydown', function(event) {
        const key = event.key;
        if (over)
            return;
        if (key === 'w' || key === 'a' || key === 's' || key === 'd' || key === 'x') {
            const direction = key === 'w' ? 'up' : key === 'a' ? 'left' : key === 's' ? 'down' : key === 'd' ? 'right' : 'wait';
            fetch(`http://localhost:8082/player/${direction}`, { method: 'PUT' })
                .catch(error => console.error('Error:', error));
        }
        
        else if(key === 'r')
        {
            displayRestartPopup();
            over = true;
        }
        // reload();
    });
});
