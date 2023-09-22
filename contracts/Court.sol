// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.9;

contract Court{

    string extract_data;

    function upload_uniqueId(string memory _extract_data) public {
        extract_data = _extract_data;
    }

    function get_uniqueId() public view returns (string memory) {
        return extract_data;
    }

    fallback() external {
       
    }


}