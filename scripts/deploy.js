const hre = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();

  console.log(`Deploying contract with address: ${deployer.address}`);

  const Court = await hre.ethers.getContractFactory("Court"); // Replace with your contract name
  const court = await Court.deploy();

  await court.waitForDeployment();

  console.log("Contract deployed to:", court.target);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
